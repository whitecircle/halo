#!/usr/bin/env bash
# Halo Competition: set up everything you need to submit, from the terminal.
#
#   bash setup.sh            asks before it creates anything
#   bash setup.sh --yes      does not ask
#
# What it does, in order. Every step is skipped if it is already done, so it is safe to run again.
#   1. Checks git and the GitHub CLI, and shows which GitHub account it will use.
#   2. Forks Halo to <you>/halo (public). Your receipts and results appear there.
#   3. Creates <you>/halo-private (private) with Halo's history. You work and submit here.
#   4. Adds .halo/submission.yaml on a branch named "work" and opens a DRAFT pull request into main.
#      The bot ignores the pull request while it is a draft.
#   5. Installs the Halo Competition bot on those two repositories. GitHub lets nobody install an app without
#      one confirmation in the browser, so this step opens a page with both repositories already selected;
#      you press Install, then Enter here.
#
# Options:  --yes  --name <private repository name>  --dir <folder to clone into>  --no-pull-request
set -euo pipefail

UPSTREAM=${HALO_UPSTREAM:-whitecircle/halo}
APP=${HALO_APP:-halo-competition}
BOT=${HALO_BOT_URL:-}   # optional: when set, the script asks the bot whether it sees both repositories
WAIT_S=${HALO_SETUP_WAIT_S:-900}

package=$(cd "$(dirname "$0")" && pwd)
yes=0; name=halo-private; dir=""; pull_request=1
while [ $# -gt 0 ]; do
  case "$1" in
    --yes) yes=1 ;;
    --name) name=$2; shift ;;
    --dir) dir=$2; shift ;;
    --no-pull-request) pull_request=0 ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done
dir=${dir:-$name}

# Colours, only on a terminal: green is what gets created and is public, pink is what is private.
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  bold=$(printf '\033[1m'); dim=$(printf '\033[2m'); plain=$(printf '\033[0m')
  green=$(printf '\033[1;32m'); pink=$(printf '\033[1;38;5;205m')
else
  bold=""; dim=""; plain=""; green=""; pink=""
fi
step() { printf '\n%s[%s/5] %s%s\n' "$bold" "$1" "$2" "$plain"; }
ok()   { printf '      %sdone:%s %s\n' "$green" "$plain" "$1"; }
skip() { printf '      %salready there: %s%s\n' "$dim" "$1" "$plain"; }
stop() { printf '\nStopped: %s\n' "$1" >&2; exit 1; }
# git over HTTPS with the GitHub CLI's sign-in, whatever credential helper this machine has
git_gh() { git -c credential.helper= -c credential.helper='!gh auth git-credential' "$@"; }

# --- 1 ---------------------------------------------------------------------------------------------------
step 1 "Checking your tools and your GitHub account"
command -v git >/dev/null || stop "git is not installed."
command -v gh >/dev/null || stop "the GitHub CLI is not installed. Get it at https://cli.github.com and run: gh auth login"
gh auth status >/dev/null 2>&1 || stop "the GitHub CLI is not signed in. Run: gh auth login"
login=$(gh api user --jq .login)
user_id=$(gh api user --jq .id)
ok "signed in to GitHub as $login"

cat <<PLAN

      This will, on the GitHub account ${bold}$login${plain}:

        ${green}create${plain}  ${green}$login/halo${plain}           public fork of $UPSTREAM; your receipts and results appear here
        ${green}create${plain}  ${pink}$login/$name${plain}   ${pink}private${plain} repository with Halo's history; you work and submit here
        ${green}create${plain}  ${pink}./$dir${plain}         a clone of the private repository, in this folder, on a branch "work"
        ${green}create${plain}  a ${pink}draft${plain} pull request in the private repository (the bot ignores it while it is a draft)
        ${bold}ask${plain}     you to install the $APP app on those two repositories, and on nothing else

      ${green}green${plain} is public, ${pink}pink${plain} is private. Nothing existing is changed or deleted.
PLAN
if [ "$yes" -ne 1 ]; then
  printf '\n      Go ahead? [y/N] '
  read -r answer
  case "$answer" in y|Y|yes) ;; *) stop "nothing was changed." ;; esac
fi

# --- 2 ---------------------------------------------------------------------------------------------------
step 2 "Your public fork of Halo"
# when the repository does not exist gh prints GitHub's error on standard output, so only its exit status is trusted
if found=$(gh api "repos/$login/halo" --jq '.parent.full_name // "none"' 2>/dev/null); then parent=$found; else parent=missing; fi
case "$parent" in
  missing)
    gh api --method POST "repos/$UPSTREAM/forks" -F default_branch_only=true --silent
    for _ in $(seq 1 60); do gh api "repos/$login/halo" --silent 2>/dev/null && break; sleep 2; done
    gh api "repos/$login/halo" --silent 2>/dev/null || stop "GitHub did not finish the fork. Run the script again in a minute."
    ok "forked to ${green}https://github.com/$login/halo${plain} (public)" ;;
  "$UPSTREAM") skip "https://github.com/$login/halo" ;;
  *) stop "$login/halo exists and is not a fork of $UPSTREAM. Rename or remove it, then run the script again." ;;
esac
[ "$(gh api "repos/$login/halo" --jq .private)" = false ] || stop "$login/halo is private. The fork has to be public."
fork_id=$(gh api "repos/$login/halo" --jq .id)

# --- 3 ---------------------------------------------------------------------------------------------------
step 3 "Your private repository"
if gh api "repos/$login/$name" --silent 2>/dev/null; then
  [ "$(gh api "repos/$login/$name" --jq .private)" = true ] || stop "$login/$name exists and is public. Choose another name with --name."
  skip "https://github.com/$login/$name"
else
  gh api --method POST user/repos -f name="$name" -F private=true -f description="Halo Competition: private work" --silent
  ok "created ${pink}https://github.com/$login/$name${plain} (private)"
fi
private_id=$(gh api "repos/$login/$name" --jq .id)

if [ -d "$dir/.git" ]; then
  skip "./$dir"
  cd "$dir"
else
  [ ! -e "$dir" ] || stop "./$dir exists and is not a git repository. Choose another folder with --dir."
  git clone --quiet "https://github.com/$UPSTREAM.git" "$dir"
  cd "$dir"
  git branch --quiet --move main 2>/dev/null || true
  ok "cloned Halo into ${pink}./$dir${plain}"
fi
git remote set-url origin "https://github.com/$login/$name.git"
git remote get-url public >/dev/null 2>&1 || git remote add public "https://github.com/$login/halo.git"
git remote get-url halo >/dev/null 2>&1 || git remote add halo "https://github.com/$UPSTREAM.git"
if [ -z "$(git_gh ls-remote --heads origin main)" ]; then
  git_gh push --quiet origin "main:refs/heads/main"
  ok "pushed Halo's history to $login/$name, branch main"
else
  skip "branch main of $login/$name"
fi

# --- 4 ---------------------------------------------------------------------------------------------------
step 4 "Your working branch and the submission file"
git_gh fetch --quiet origin main
base=$(git rev-parse FETCH_HEAD)
if git rev-parse --verify --quiet work >/dev/null; then git checkout --quiet work; else git checkout --quiet -b work FETCH_HEAD; fi
# The file ships with Halo, so a fresh copy already has it with an empty base_commit: fill that in. When it is
# missing (an older Halo), take it from beside this script.
[ -f .halo/submission.yaml ] || { mkdir -p .halo; cp "$package/submission.yaml" .halo/submission.yaml; }
if grep -q '^base_commit: ""' .halo/submission.yaml; then
  sed "s/^base_commit: \"\"/base_commit: \"$base\"/" .halo/submission.yaml > .halo/submission.yaml.new
  mv .halo/submission.yaml.new .halo/submission.yaml
  git add .halo/submission.yaml
  git -c user.name="$login" -c user.email="$user_id+$login@users.noreply.github.com" -c commit.gpgsign=false \
    commit --quiet -m "Set base_commit in the submission file"
  ok "set base_commit in .halo/submission.yaml on branch work (Halo $(printf %.12s "$base"))"
else
  skip ".halo/submission.yaml"
fi
git_gh push --quiet --set-upstream origin work
pull_url=""
if [ "$pull_request" -eq 1 ]; then
  pull_url=$(gh pr list --repo "$login/$name" --head work --state open --json url --jq '.[0].url // ""')
  if [ -n "$pull_url" ]; then
    skip "pull request $pull_url"
  else
    pull_url=$(gh pr create --repo "$login/$name" --base main --head work --draft \
      --title "Halo Competition submission" \
      --body "Nothing is submitted while this pull request is a draft. Fill .halo/submission.yaml, push your work, then mark it ready for review to submit.")
    ok "opened a ${pink}draft${plain} pull request: ${pink}$pull_url${plain}"
  fi
fi

# --- 5 ---------------------------------------------------------------------------------------------------
step 5 "Installing the Halo Competition bot on the two repositories"
status() { [ -n "$BOT" ] && curl --silent --max-time 20 "$BOT/api/setup/$login" 2>/dev/null || true; }
field() { printf '%s' "$1" | sed -n "s/.*\"$2\":\"\\([^\"]*\\)\".*/\\1/p"; }
ready() { case "$1" in *'"ready":true'*) return 0 ;; *) return 1 ;; esac; }

seen=$(status)
if ready "$seen"; then
  skip "the bot has $login/halo and your private repository"
else
  install="https://github.com/apps/$APP/installations/new/permissions?suggested_target_id=$user_id&repository_ids[]=$fork_id&repository_ids[]=$private_id"
  cat <<INSTALL

      GitHub asks for one confirmation in the browser. Open this page; both repositories are already
      selected ("Only select repositories": ${green}$login/halo${plain} and ${pink}$login/$name${plain}). Press ${bold}Install${plain}.

        $install

      If the app is already on your account, the page is its settings: add the two repositories and Save.
INSTALL
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$install" >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then open "$install" >/dev/null 2>&1 || true; fi
  if [ -z "$BOT" ]; then
    printf '\n      Press Enter here when you have pressed Install in the browser. '
    read -r _
    seen='"ready":true'
    ok "installed on ${green}$login/halo${plain} and ${pink}$login/$name${plain}, as you confirmed"
  else
  printf '\n      Waiting for the bot to see them (up to %s minutes; Ctrl-C to stop, run again later to resume)\n' "$((WAIT_S / 60))"
  said=""; waited=0
  until ready "$seen"; do
    message=$(field "$seen" message)
    if [ -n "$message" ] && [ "$message" != "$said" ]; then printf '      %s...%s %s\n' "$dim" "$plain" "$message"; said=$message; fi
    [ "$waited" -lt "$WAIT_S" ] || stop "the bot does not see both repositories yet. Finish the installation in the browser and run the script again."
    sleep 5; waited=$((waited + 5))
    seen=$(status)
  done
  ok "the bot has $(field "$seen" public_fork) and your private repository"
  fi
fi

submit_step="gh pr create --base main --head work"
[ -z "$pull_url" ] || submit_step="gh pr ready $pull_url"
cat <<DONE

${green}Ready.${plain} You are in ./$dir on the branch "work".

  Work:     commit and push to "work" as often as you like. Nothing is read while the pull request is a draft.
  Submit:   1. fill .halo/submission.yaml
            2. git push
            3. $submit_step
            The bot answers on the pull request within a minute with your submission ID and arrival time.
  Resubmit: every later push to that pull request is a new submission.
DONE
