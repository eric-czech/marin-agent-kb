---
name: clone-marin-branch
description: Clone marin-community/marin onto a new branch at repos/marin-br/<slug>.
argument-hint: <branch-name> <originating-branch>
allowed-tools: [Bash]
---

# clone-marin-branch

`$ARGUMENTS` → `<branch-name> <originating-branch>`. If either is missing, ask — there
is no default originating branch, be explicit. Let `BR=<branch-name>`, `SRC=<originating-branch>`.

**Slug:** flatten `BR` into a directory name — replace `/`, filesystem-unsafe chars
(`\:*?"<>|`) and whitespace with `-`, collapse repeated `-`, trim leading/trailing `-`.
Then `SLUG=<sanitized BR>` and `DEST=repos/marin-br/$SLUG`.

## Steps

```bash
[ -e "$DEST" ] && { echo "refusing to overwrite $DEST"; exit 1; }    # never overwrite
mkdir -p repos/marin-br
git clone --branch "$SRC" --origin origin git@github.com:marin-community/marin.git "$DEST"
git -C "$DEST" checkout -b "$BR"      # original BR, not SLUG — git refs keep slashes
```

Clone is over SSH (see the GitHub SSH key step in `skills/setup-dev-vm.md`).

## Hand-off

Stop here and report the path to the new branch checkout (`$DEST`).
