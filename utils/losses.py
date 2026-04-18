import torch

# uses real units
def init_brt_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor, use_vhat_guidance=False):
    def brt_hjivi_loss(
        state,
        value,
        dvdt,
        dvds,
        boundary_value,
        dirichlet_mask,
        output,
        guidance_mask=None,
        guidance_vhat=None,
        guidance_weight=None,
    ):
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds)
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(
                    diff_constraint_hom, value - boundary_value)
        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
        losses = {}
        if dynamics.deepreach_model == 'exact':
            if torch.all(dirichlet_mask):
                # pretraining
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                losses['diff_constraint_hom'] = torch.abs(diff_constraint_hom).sum()
                if use_vhat_guidance and guidance_mask is not None and guidance_vhat is not None and guidance_weight is not None:
                    active = guidance_mask > 0.5
                    if torch.any(active):
                        vhat_abs_err = torch.abs(value[active] - guidance_vhat[active])
                        losses['vhat_guidance'] = torch.sum(guidance_weight[active] * vhat_abs_err)
                return losses

        losses['dirichlet'] = torch.abs(dirichlet).sum() / dirichlet_loss_divisor
        losses['diff_constraint_hom'] = torch.abs(diff_constraint_hom).sum()
        if use_vhat_guidance and guidance_mask is not None and guidance_vhat is not None and guidance_weight is not None:
            active = guidance_mask > 0.5
            if torch.any(active):
                vhat_abs_err = torch.abs(value[active] - guidance_vhat[active])
                losses['vhat_guidance'] = torch.sum(guidance_weight[active] * vhat_abs_err)
        return losses

    return brt_hjivi_loss
def init_brat_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor, use_vhat_guidance=False):
    def brat_hjivi_loss(
        state,
        value,
        dvdt,
        dvds,
        boundary_value,
        reach_value,
        avoid_value,
        dirichlet_mask,
        output,
        guidance_mask=None,
        guidance_vhat=None,
        guidance_weight=None,
    ):
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds)
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.min(
                    torch.max(diff_constraint_hom, value - reach_value), value + avoid_value)

        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
        losses = {}
        if dynamics.deepreach_model == 'exact':
            if torch.all(dirichlet_mask):
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                losses['diff_constraint_hom'] = torch.abs(diff_constraint_hom).sum()
                if use_vhat_guidance and guidance_mask is not None and guidance_vhat is not None and guidance_weight is not None:
                    active = guidance_mask > 0.5
                    if torch.any(active):
                        vhat_abs_err = torch.abs(value[active] - guidance_vhat[active])
                        losses['vhat_guidance'] = torch.sum(guidance_weight[active] * vhat_abs_err)
                return losses
        losses['dirichlet'] = torch.abs(dirichlet).sum() / dirichlet_loss_divisor
        losses['diff_constraint_hom'] = torch.abs(diff_constraint_hom).sum()
        if use_vhat_guidance and guidance_mask is not None and guidance_vhat is not None and guidance_weight is not None:
            active = guidance_mask > 0.5
            if torch.any(active):
                vhat_abs_err = torch.abs(value[active] - guidance_vhat[active])
                losses['vhat_guidance'] = torch.sum(guidance_weight[active] * vhat_abs_err)
        return losses
    return brat_hjivi_loss
