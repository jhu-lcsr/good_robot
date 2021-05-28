import torch
from utils.simple_profiler import SimpleProfiler

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

# Beta
FixedBeta = torch.distributions.Beta
log_prob_beta = FixedBeta.log_prob
FixedBeta.log_probs = lambda self, actions: log_prob_beta(
    self, actions).sum(
        -1, keepdim=True)

beta_entropy = FixedBeta.entropy
FixedBeta.entropy = lambda self: beta_entropy(self).mean(-1)
#FixedBeta.mode = lambda self: self.mean
# TODO: This is the mode only if alpha > 1 and beta > 1. Otherwise it's the anti-mode.
FixedBeta.mode = lambda self: (self.concentration0 - 1) / (self.concentration0 + self.concentration1 - 2)

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions).sum(
        -1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).mean(-1)

FixedNormal.mode = lambda self: self.mean


# Bernoulli
FixedBernoulli = torch.distributions.Bernoulli

log_prob_bernoulli = FixedBernoulli.log_prob
FixedBernoulli.log_probs = lambda self, actions: log_prob_bernoulli(
    self, actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

bernoulli_entropy = FixedBernoulli.entropy
FixedBernoulli.entropy = lambda self: bernoulli_entropy(self).mean(-1)
FixedBernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()

PROFILE=True

# RestrictedGaussian
class BoundedNormal(torch.distributions.normal.Normal):
    def __init__(self, mean, std, min, max):
        # Clamp the mean to not go more than one std dev outside of the permitted interval
        #mean = torch.max(torch.min(mean, max + std), min - std)
        super(BoundedNormal, self).__init__(mean, std)
        self.lower = min
        self.upper = max
        self.proposal = torch.distributions.Uniform(self.lower, self.upper)
        self.uniform = torch.distributions.Uniform(0, 1)

    def log_prob(self, value):
        interval_prob = self.cdf(self.upper) - self.cdf(self.lower)
        interval_log_prob = torch.log(interval_prob).detach()
        logprob = super(BoundedNormal, self).log_prob(value)
        # P(X) = P(X) under gaussian / P(interval)
        return logprob - interval_log_prob

    def entropy(self):
        return super(BoundedNormal, self).entropy().mean()

    def sample(self, sample_shape=torch.Size([])):
        # Rejection sampling from a bounded interval
        dim = self.mean.shape[0]
        #self.prof.tick("out")

        proposals = self.proposal.sample(self.mean.shape).to(self.mean.device)
        uniforms = self.uniform.sample(self.mean.shape).to(self.mean.device)
        pdfs = torch.exp(self.log_prob(proposals))
        maxpdf = torch.exp(self.log_prob(torch.clamp(self.mean, self.lower, self.upper)))
        norm_pdfs = pdfs / maxpdf

        accepted_samples = torch.zeros_like(proposals)
        overall_accept = uniforms < norm_pdfs
        accepted_samples[overall_accept] = proposals[overall_accept]

        count = 0
        while overall_accept.long().sum() < dim:
            proposals = self.proposal.sample(self.mean.shape).to(self.mean.device)
            uniforms = self.uniform.sample(self.mean.shape).to(self.mean.device)
            pdfs = torch.exp(self.log_prob(proposals))
            norm_pdfs = pdfs / maxpdf
            accept = uniforms < norm_pdfs
            # TODO: Don't re-sample already accepted values
            new_accept = accept and not overall_accept
            accepted_samples[new_accept] = proposals[new_accept]
            overall_accept = overall_accept or accept
            count += 1

        if count > 10:
            print(f"Sampling took {count} attempts!")
        if count > 100:
            print("Shite!")
        #self.prof.tick("sampling")
        #self.prof.loop()
        #self.prof.print_stats(1)
        return accepted_samples

    def mode(self):
        return self.mean

