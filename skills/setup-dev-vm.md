---
name: setup-dev-vm
description: Bootstrap a fresh Ubuntu VM for marin agent work — gcloud/SA, GitHub auth, env, skills.
---

# Set Up a Dev VM

From scratch on a fresh Ubuntu x86_64 VM: gcloud, GitHub token + SSH key,
`~/marin.env`, and agent skills.

## gcloud + service-account key

Reuse the **existing** `eczech-agent@hai-gcp-models` key — GCP can't re-download key
material, so never mint a new one casually. (If the SA / key doesn't exist yet,
create it first — see `docs/setup-service-account.md`.)

```bash
curl -fsSL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz | tar -xz -C ~
~/google-cloud-sdk/install.sh -q --path-update=true
mkdir -p ~/.config/gcloud/keys && mv ~/eczech-agent.json ~/.config/gcloud/keys/
chmod 700 ~/.config/gcloud/keys && chmod 600 ~/.config/gcloud/keys/eczech-agent.json
KEY=~/.config/gcloud/keys/eczech-agent.json
gcloud auth activate-service-account --key-file="$KEY"
gcloud config set project hai-gcp-models
echo "export GOOGLE_APPLICATION_CREDENTIALS=$KEY" >> ~/.bashrc
```

## GitHub: fine-grained PAT + SSH key

PAT scopes: Contents RW, Pull requests RW, Metadata R (+ Gists RW, Git SSH keys RW).

```bash
printf 'export GITHUB_TOKEN=%s\nexport GH_TOKEN=$GITHUB_TOKEN\n' 'PASTE_TOKEN' > ~/.config/gh-token.env
chmod 600 ~/.config/gh-token.env
echo '[ -f ~/.config/gh-token.env ] && source ~/.config/gh-token.env' >> ~/.bashrc
# install gh to ~/.local/bin, then:
gh auth setup-git && gh api user --jq .login        # verify -> your login

ssh-keygen -t ed25519 -C "$(whoami)@vm" -f ~/.ssh/id_ed25519 -N ""
chmod 700 ~/.ssh; chmod 600 ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub      # add at github.com/settings/ssh/new
ssh -T git@github.com          # verify -> "Hi <user>!"
```

## ~/marin.env

Project config + secrets (`chmod 600`); source it from `~/.bashrc`.

```bash
export USERNAME=eczech
export PROJECT_ID=hai-gcp-models
export HUGGING_FACE_HUB_TOKEN=hf_...   ; export HF_TOKEN=$HUGGING_FACE_HUB_TOKEN
export WANDB_API_KEY=...               ; export WANDB_PROJECT=marin-dna
export WANDB_ENTITY=eric-czech
```

## Agent skills

Claude Code reads `~/.claude/skills/<name>/SKILL.md`; Codex reads
`~/.codex/prompts/<name>.md`. Neither reads `~/.agents`. Keep ONE canonical copy in
`~/.agents` and symlink it into each agent:

```bash
ln -s ~/.agents/skills ~/.claude/skills                          # whole dir (or per-entry if it exists)
ln -s ~/.agents/skills/clone-marin-branch/SKILL.md ~/.codex/prompts/clone-marin-branch.md
```
