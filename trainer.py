"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen, AdaINGen_double
from utils import (
    weights_init,
    get_model_list,
    vgg_preprocess,
    load_vgg16,
    get_scheduler,
)
from torch.autograd import Variable
import torch
import torch.nn as nn
import os


class DoubleMUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters, comet_exp=None):
        super(MUNIT_Trainer, self).__init__()
        self.comet_exp = comet_exp
        lr = hyperparameters["lr"]
        # Initiate the networks
        self.gen = AdaINGen_double(
            hyperparameters["input_dim_a"], hyperparameters["gen"]
        )
        self.randn_shift = hyperparameters.get("randn_shift", 2)

        # self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        # self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b

        self.dis_a = MsImageDis(
            hyperparameters["input_dim_a"], hyperparameters["dis"]
        )  # discriminator for domain a
        self.dis_b = MsImageDis(
            hyperparameters["input_dim_b"], hyperparameters["dis"]
        )  # discriminator for domain b

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters["gen"]["style_dim"]

        # fix the noise used in sampling
        display_size = int(hyperparameters["display_size"])
        self.s_a = self.shifted_randn(
            display_size, self.style_dim, 1, 1, domain="a"
        ).cuda()
        self.s_b = self.shifted_randn(
            display_size, self.style_dim, 1, 1, domain="b"
        ).cuda()

        # Setup the optimizers
        beta1 = hyperparameters["beta1"]
        beta2 = hyperparameters["beta2"]
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen.parameters())  # + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam(
            [p for p in dis_params if p.requires_grad],
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=hyperparameters["weight_decay"],
        )
        self.gen_opt = torch.optim.Adam(
            [p for p in gen_params if p.requires_grad],
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=hyperparameters["weight_decay"],
        )
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters["init"]))
        self.dis_a.apply(weights_init("gaussian"))
        self.dis_b.apply(weights_init("gaussian"))

        # Load VGG model if needed
        if "vgg_w" in hyperparameters.keys() and hyperparameters["vgg_w"] > 0:
            self.vgg = load_vgg16(hyperparameters["vgg_model_path"] + "/models")
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def disantanglement_criterion(self, s_a_prime, s_b_prime):
        return torch.sigmoid(torch.norm(s_a_prime.mean(0) - s_b_prime.mean(0)))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen.encode(x_a, 1)
        c_b, s_b_fake = self.gen.encode(x_b, 2)
        x_ba = self.gen.decode(c_b, s_a, 1)
        x_ab = self.gen.decode(c_a, s_b, 2)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        s_a = Variable(
            self.shifted_randn(x_a.size(0), self.style_dim, 1, 1, domain="a").cuda()
        )
        s_b = Variable(
            self.shifted_randn(x_b.size(0), self.style_dim, 1, 1, domain="b").cuda()
        )
        # encode
        c_a, self.s_a_prime = self.gen.encode(x_a, 1)

        c_b, self.s_b_prime = self.gen.encode(x_b, 2)

        # decode (within domain)
        x_a_recon = self.gen.decode(c_a, self.s_a_prime, 1)
        x_b_recon = self.gen.decode(c_b, self.s_b_prime, 2)

        # decode (cross domain)
        x_ba = self.gen.decode(c_b, s_a, 1)
        x_ab = self.gen.decode(c_a, s_b, 2)

        # encode again
        c_b_recon, s_a_recon = self.gen.encode(x_ba, 1)
        c_a_recon, s_b_recon = self.gen.encode(x_ab, 2)

        # decode again (if needed)
        x_aba = (
            self.gen.decode(c_a_recon, self.s_a_prime, 1)
            if hyperparameters["recon_x_cyc_w"] > 0
            else None
        )
        x_bab = (
            self.gen.decode(c_b_recon, self.s_b_prime, 2)
            if hyperparameters["recon_x_cyc_w"] > 0
            else None
        )

        # disentanglement loss
        self.disen_loss_ab = self.disantanglement_criterion(
            self.s_a_prime, self.s_b_prime
        )

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = (
            self.recon_criterion(x_aba, x_a)
            if hyperparameters["recon_x_cyc_w"] > 0
            else 0
        )
        self.loss_gen_cycrecon_x_b = (
            self.recon_criterion(x_bab, x_b)
            if hyperparameters["recon_x_cyc_w"] > 0
            else 0
        )

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = (
            self.compute_vgg_loss(self.vgg, x_ba, x_b)
            if hyperparameters["vgg_w"] > 0
            else 0
        )
        self.loss_gen_vgg_b = (
            self.compute_vgg_loss(self.vgg, x_ab, x_a)
            if hyperparameters["vgg_w"] > 0
            else 0
        )

        # total loss
        self.loss_gen_total = (
            hyperparameters["gan_w"] * self.loss_gen_adv_a
            + hyperparameters["gan_w"] * self.loss_gen_adv_b
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_a
            + hyperparameters["recon_s_w"] * self.loss_gen_recon_s_a
            + hyperparameters["recon_c_w"] * self.loss_gen_recon_c_a
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_b
            + hyperparameters["recon_s_w"] * self.loss_gen_recon_s_b
            + hyperparameters["recon_c_w"] * self.loss_gen_recon_c_b
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cycrecon_x_a
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cycrecon_x_b
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_a
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_b
            + hyperparameters["disen_ab"] * self.disen_loss_ab
        )
        # +\
        # 1.0  * self.disantangle_style_loss
        self.comet_exp.log_metric("loss_gen_total", self.loss_gen_total)
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean(
            (self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2
        )

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(
            self.shifted_randn(x_a.size(0), self.style_dim, 1, 1, domain="a").cuda()
        )
        s_b2 = Variable(
            self.shifted_randn(x_b.size(0), self.style_dim, 1, 1, domain="b").cuda()
        )
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen.encode(x_a[i].unsqueeze(0), 1)
            c_b, s_b_fake = self.gen.encode(x_b[i].unsqueeze(0), 2)
            x_a_recon.append(self.gen.decode(c_a, s_a_fake, 1))
            x_b_recon.append(self.gen.decode(c_b, s_b_fake, 2))
            x_ba1.append(self.gen.decode(c_b, s_a1[i].unsqueeze(0), 1))
            x_ba2.append(self.gen.decode(c_b, s_a2[i].unsqueeze(0), 1))
            x_ab1.append(self.gen.decode(c_a, s_b1[i].unsqueeze(0), 2))
            x_ab2.append(self.gen.decode(c_a, s_b2[i].unsqueeze(0), 2))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        s_a = Variable(
            self.shifted_randn(x_a.size(0), self.style_dim, 1, 1, domain="a").cuda()
        )
        s_b = Variable(
            self.shifted_randn(x_b.size(0), self.style_dim, 1, 1, domain="b").cuda()
        )
        # encode
        c_a, _ = self.gen.encode(x_a, 1)
        c_b, _ = self.gen.encode(x_b, 2)
        # decode (cross domain)
        x_ba = self.gen.decode(c_b, s_a, 1)
        x_ab = self.gen.decode(c_a, s_b, 2)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = (
            hyperparameters["gan_w"] * self.loss_dis_a
            + hyperparameters["gan_w"] * self.loss_dis_b
        )
        self.loss_dis_total.backward()
        self.dis_opt.step()
        self.comet_exp.log_metric("loss_dis_total", self.loss_dis_total)

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict["2"])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict["a"])
        self.dis_b.load_state_dict(state_dict["b"])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        self.dis_opt.load_state_dict(state_dict["dis"])
        self.gen_opt.load_state_dict(state_dict["gen"])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print("Resume from iteration %d" % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, "gen_%08d.pt" % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, "dis_%08d.pt" % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, "optimizer.pt")
        torch.save({"2": self.gen.state_dict()}, gen_name)
        torch.save(
            {"a": self.dis_a.state_dict(), "b": self.dis_b.state_dict()}, dis_name
        )
        torch.save(
            {"gen": self.gen_opt.state_dict(), "dis": self.dis_opt.state_dict()},
            opt_name,
        )

    def shifted_randn(self, *sizes, domain="a"):
        return (
            torch.randn(*sizes) + self.randn_shift
            if domain == "a"
            else torch.randn(*sizes) - self.randn_shift
        )


class MUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters, comet_exp=None):
        super(MUNIT_Trainer, self).__init__()

        self.comet_exp = comet_exp
        self.randn_shift = hyperparameters.get("randn_shift", 2)

        print("self.randn_shift", self.randn_shift)

        lr = hyperparameters["lr"]
        # Initiate the networks
        self.gen_a = AdaINGen(
            hyperparameters["input_dim_a"], hyperparameters["gen"]
        )  # auto-encoder for domain a
        self.gen_b = AdaINGen(
            hyperparameters["input_dim_b"], hyperparameters["gen"]
        )  # auto-encoder for domain b
        self.dis_a = MsImageDis(
            hyperparameters["input_dim_a"], hyperparameters["dis"]
        )  # discriminator for domain a
        self.dis_b = MsImageDis(
            hyperparameters["input_dim_b"], hyperparameters["dis"]
        )  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters["gen"]["style_dim"]

        # fix the noise used in sampling
        display_size = int(hyperparameters["display_size"])
        self.s_a = self.shifted_randn(
            display_size, self.style_dim, 1, 1, domain="a"
        ).cuda()
        self.s_b = self.shifted_randn(
            display_size, self.style_dim, 1, 1, domain="b"
        ).cuda()

        # Setup the optimizers
        beta1 = hyperparameters["beta1"]
        beta2 = hyperparameters["beta2"]
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam(
            [p for p in dis_params if p.requires_grad],
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=hyperparameters["weight_decay"],
        )
        self.gen_opt = torch.optim.Adam(
            [p for p in gen_params if p.requires_grad],
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=hyperparameters["weight_decay"],
        )
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters["init"]))
        self.dis_a.apply(weights_init("gaussian"))
        self.dis_b.apply(weights_init("gaussian"))

        # self.style_store = {
        #     "a": [
        #         torch.randn(display_size, self.style_dim, 1, 1).cuda()
        #         for _ in range(hyperparameters["style_store_size"] or 100)
        #     ],
        #     "b": [
        #         torch.randn(display_size, self.style_dim, 1, 1).cuda()
        #         for _ in range(hyperparameters["style_store_size"] or 100)
        #     ],
        # }

        # Load VGG model if needed
        if "vgg_w" in hyperparameters.keys() and hyperparameters["vgg_w"] > 0:
            self.vgg = load_vgg16(hyperparameters["vgg_model_path"] + "/models")
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def disantanglement_criterion(self, s_a_prime, s_b_prime):
        return torch.sigmoid(torch.norm(s_a_prime.mean(0) - s_b_prime.mean(0)))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        s_a = Variable(
            self.shifted_randn(x_a.size(0), self.style_dim, 1, 1, domain="a").cuda()
        )
        s_b = Variable(
            self.shifted_randn(x_b.size(0), self.style_dim, 1, 1, domain="b").cuda()
        )
        # encode
        c_a, self.s_a_prime = self.gen_a.encode(x_a)
        c_b, self.s_b_prime = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, self.s_a_prime)
        x_b_recon = self.gen_b.decode(c_b, self.s_b_prime)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # encode again
        c_b_recon, s_a_recon = self.gen_a.encode(x_ba)
        c_a_recon, s_b_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = (
            self.gen_a.decode(c_a_recon, self.s_a_prime)
            if hyperparameters["recon_x_cyc_w"] > 0
            else None
        )
        x_bab = (
            self.gen_b.decode(c_b_recon, self.s_b_prime)
            if hyperparameters["recon_x_cyc_w"] > 0
            else None
        )

        # disentanglement loss
        self.disen_loss_ab = self.disantanglement_criterion(
            self.s_a_prime, self.s_b_prime
        )

        # reconstruction loss
        # torch.mean(torch.abs(input - target))
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_cycrecon_x_a = (
            self.recon_criterion(x_aba, x_a)
            if hyperparameters["recon_x_cyc_w"] > 0
            else 0
        )
        self.loss_gen_cycrecon_x_b = (
            self.recon_criterion(x_bab, x_b)
            if hyperparameters["recon_x_cyc_w"] > 0
            else 0
        )
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = (
            self.compute_vgg_loss(self.vgg, x_ba, x_b)
            if hyperparameters["vgg_w"] > 0
            else 0
        )
        self.loss_gen_vgg_b = (
            self.compute_vgg_loss(self.vgg, x_ab, x_a)
            if hyperparameters["vgg_w"] > 0
            else 0
        )
        # total loss
        # print('\n\n', hyperparameters, "\n\n")
        self.loss_gen_total = (
            hyperparameters["gan_w"] * self.loss_gen_adv_a
            + hyperparameters["gan_w"] * self.loss_gen_adv_b
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_a
            + hyperparameters["recon_s_w"] * self.loss_gen_recon_s_a
            + hyperparameters["recon_c_w"] * self.loss_gen_recon_c_a
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_b
            + hyperparameters["recon_s_w"] * self.loss_gen_recon_s_b
            + hyperparameters["recon_c_w"] * self.loss_gen_recon_c_b
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cycrecon_x_a
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cycrecon_x_b
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_a
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_b
            + hyperparameters["disen_ab"] * self.disen_loss_ab
        )
        self.comet_exp.log_metric("loss_gen_total", self.loss_gen_total)
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean(
            (self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2
        )

    def shifted_randn(self, *sizes, domain="a"):
        return (
            torch.randn(*sizes) + self.randn_shift
            if domain == "a"
            else torch.randn(*sizes) - self.randn_shift
        )

    def sample(self, x_a, x_b):
        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(
            self.shifted_randn(x_a.size(0), self.style_dim, 1, 1, domain="a").cuda()
        )
        s_b2 = Variable(
            self.shifted_randn(x_b.size(0), self.style_dim, 1, 1, domain="b").cuda()
        )
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake))
            x_ba1.append(self.gen_a.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen_a.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen_b.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen_b.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        s_a = Variable(
            self.shifted_randn(x_a.size(0), self.style_dim, 1, 1, domain="a").cuda()
        )
        s_b = Variable(
            self.shifted_randn(x_b.size(0), self.style_dim, 1, 1, domain="b").cuda()
        )
        # encode
        c_a, _ = self.gen_a.encode(x_a)
        c_b, _ = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = (
            hyperparameters["gan_w"] * self.loss_dis_a
            + hyperparameters["gan_w"] * self.loss_dis_b
        )
        self.comet_exp.log_metric("loss_dis_total", self.loss_dis_total)
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict["a"])
        self.gen_b.load_state_dict(state_dict["b"])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict["a"])
        self.dis_b.load_state_dict(state_dict["b"])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        self.dis_opt.load_state_dict(state_dict["dis"])
        self.gen_opt.load_state_dict(state_dict["gen"])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print("Resume from iteration %d" % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, "gen_%08d.pt" % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, "dis_%08d.pt" % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, "optimizer.pt")
        torch.save(
            {"a": self.gen_a.state_dict(), "b": self.gen_b.state_dict()}, gen_name
        )
        torch.save(
            {"a": self.dis_a.state_dict(), "b": self.dis_b.state_dict()}, dis_name
        )
        torch.save(
            {"gen": self.gen_opt.state_dict(), "dis": self.dis_opt.state_dict()},
            opt_name,
        )


class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters["lr"]
        # Initiate the networks
        self.gen_a = VAEGen(
            hyperparameters["input_dim_a"], hyperparameters["gen"]
        )  # auto-encoder for domain a
        self.gen_b = VAEGen(
            hyperparameters["input_dim_b"], hyperparameters["gen"]
        )  # auto-encoder for domain b
        self.dis_a = MsImageDis(
            hyperparameters["input_dim_a"], hyperparameters["dis"]
        )  # discriminator for domain a
        self.dis_b = MsImageDis(
            hyperparameters["input_dim_b"], hyperparameters["dis"]
        )  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        # Setup the optimizers
        beta1 = hyperparameters["beta1"]
        beta2 = hyperparameters["beta2"]
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam(
            [p for p in dis_params if p.requires_grad],
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=hyperparameters["weight_decay"],
        )
        self.gen_opt = torch.optim.Adam(
            [p for p in gen_params if p.requires_grad],
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=hyperparameters["weight_decay"],
        )
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters["init"]))
        self.dis_a.apply(weights_init("gaussian"))
        self.dis_b.apply(weights_init("gaussian"))

        # Load VGG model if needed
        if "vgg_w" in hyperparameters.keys() and hyperparameters["vgg_w"] > 0:
            self.vgg = load_vgg16(hyperparameters["vgg_model_path"] + "/models")
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (within domain)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = (
            self.gen_a.decode(h_a_recon + n_a_recon)
            if hyperparameters["recon_x_cyc_w"] > 0
            else None
        )
        x_bab = (
            self.gen_b.decode(h_b_recon + n_b_recon)
            if hyperparameters["recon_x_cyc_w"] > 0
            else None
        )

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = (
            self.compute_vgg_loss(self.vgg, x_ba, x_b)
            if hyperparameters["vgg_w"] > 0
            else 0
        )
        self.loss_gen_vgg_b = (
            self.compute_vgg_loss(self.vgg, x_ab, x_a)
            if hyperparameters["vgg_w"] > 0
            else 0
        )
        # total loss
        self.loss_gen_total = (
            hyperparameters["gan_w"] * self.loss_gen_adv_a
            + hyperparameters["gan_w"] * self.loss_gen_adv_b
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_a
            + hyperparameters["recon_kl_w"] * self.loss_gen_recon_kl_a
            + hyperparameters["recon_x_w"] * self.loss_gen_recon_x_b
            + hyperparameters["recon_kl_w"] * self.loss_gen_recon_kl_b
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cyc_x_a
            + hyperparameters["recon_kl_cyc_w"] * self.loss_gen_recon_kl_cyc_aba
            + hyperparameters["recon_x_cyc_w"] * self.loss_gen_cyc_x_b
            + hyperparameters["recon_kl_cyc_w"] * self.loss_gen_recon_kl_cyc_bab
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_a
            + hyperparameters["vgg_w"] * self.loss_gen_vgg_b
        )
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean(
            (self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2
        )

    def sample(self, x_a, x_b):
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross domain)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = (
            hyperparameters["gan_w"] * self.loss_dis_a
            + hyperparameters["gan_w"] * self.loss_dis_b
        )
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict["a"])
        self.gen_b.load_state_dict(state_dict["b"])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict["a"])
        self.dis_b.load_state_dict(state_dict["b"])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        self.dis_opt.load_state_dict(state_dict["dis"])
        self.gen_opt.load_state_dict(state_dict["gen"])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print("Resume from iteration %d" % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, "gen_%08d.pt" % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, "dis_%08d.pt" % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, "optimizer.pt")
        torch.save(
            {"a": self.gen_a.state_dict(), "b": self.gen_b.state_dict()}, gen_name
        )
        torch.save(
            {"a": self.dis_a.state_dict(), "b": self.dis_b.state_dict()}, dis_name
        )
        torch.save(
            {"gen": self.gen_opt.state_dict(), "dis": self.dis_opt.state_dict()},
            opt_name,
        )

