#!/usr/bin/env bash
# Sets up a Halo Competition participant: a public fork of Halo, a private copy to work in, and the
# competition bot installed on both. Safe to run again.
#
#   bash setup.sh [--yes] [--name <private repository>] [--dir <folder>]
set -euo pipefail

UPSTREAM=${HALO_UPSTREAM:-whitecircle/halo}
APP=${HALO_APP:-halo-competition}

package=$(cd "$(dirname "$0")" && pwd)
yes=0; name=halo-private; dir=""
while [ $# -gt 0 ]; do
  case "$1" in
    --yes) yes=1 ;;
    --name) name=$2; shift ;;
    --dir) dir=$2; shift ;;
    -h|--help) sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done
dir=${dir:-$name}

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  bold=$(printf '\033[1m'); dim=$(printf '\033[2m'); plain=$(printf '\033[0m')
  green=$(printf '\033[1;32m'); pink=$(printf '\033[1;38;5;205m')
else
  bold=""; dim=""; plain=""; green=""; pink=""
fi
step() { printf '\n%s[%s/4] %s%s\n' "$bold" "$1" "$2" "$plain"; }
ok()   { printf '      %sdone:%s %s\n' "$green" "$plain" "$1"; }
skip() { printf '      %salready there: %s%s\n' "$dim" "$1" "$plain"; }
stop() { printf '\nStopped: %s\n' "$1" >&2; exit 1; }
git_gh() { git -c credential.helper= -c credential.helper='!gh auth git-credential' "$@"; }

step 1 "Your GitHub account"
command -v git >/dev/null || stop "git is not installed."
command -v gh >/dev/null || stop "the GitHub CLI is not installed. Get it at https://cli.github.com and run: gh auth login"
gh auth status >/dev/null 2>&1 || stop "the GitHub CLI is not signed in. Run: gh auth login"
login=$(gh api user --jq .login)
user_id=$(gh api user --jq .id)
ok "signed in as $login"

cat <<PLAN

      This creates, on the GitHub account ${bold}$login${plain}:

        ${green}$login/halo${plain}           public fork of $UPSTREAM, where your results appear
        ${pink}$login/$name${plain}   private repository with Halo's history, where you work and submit
        ${pink}./$dir${plain}         a clone of the private repository, on a branch "work"

      ${green}green${plain} is public, ${pink}pink${plain} is private. Then it asks you to install the $APP app on the two
      repositories. Nothing existing is changed or deleted.
PLAN
if [ "$yes" -ne 1 ]; then
  printf '\n      Go ahead? [y/N] '
  read -r answer
  case "$answer" in y|Y|yes) ;; *) stop "nothing was changed." ;; esac
fi

step 2 "Public fork"
# gh prints GitHub's error on standard output when the repository is missing, so only its exit status counts
if found=$(gh api "repos/$login/halo" --jq '.parent.full_name // "none"' 2>/dev/null); then parent=$found; else parent=missing; fi
case "$parent" in
  missing)
    gh api --method POST "repos/$UPSTREAM/forks" -F default_branch_only=true --silent
    for _ in $(seq 1 60); do gh api "repos/$login/halo" --silent 2>/dev/null && break; sleep 2; done
    gh api "repos/$login/halo" --silent 2>/dev/null || stop "GitHub did not finish the fork. Run the script again in a minute."
    ok "forked to ${green}https://github.com/$login/halo${plain}" ;;
  "$UPSTREAM") skip "https://github.com/$login/halo" ;;
  *) stop "$login/halo exists and is not a fork of $UPSTREAM. Rename or remove it, then run the script again." ;;
esac
[ "$(gh api "repos/$login/halo" --jq .private)" = false ] || stop "$login/halo is private. The fork has to be public."
fork_id=$(gh api "repos/$login/halo" --jq .id)

step 3 "Private repository"
if gh api "repos/$login/$name" --silent 2>/dev/null; then
  [ "$(gh api "repos/$login/$name" --jq .private)" = true ] || stop "$login/$name exists and is public. Choose another name with --name."
  skip "https://github.com/$login/$name"
else
  gh api --method POST user/repos -f name="$name" -F private=true -f description="Halo Competition: private work" --silent
  ok "created ${pink}https://github.com/$login/$name${plain}"
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
  ok "pushed Halo's history to $login/$name"
else
  skip "branch main of $login/$name"
fi

git_gh fetch --quiet origin main
base=$(git rev-parse FETCH_HEAD)
if git rev-parse --verify --quiet work >/dev/null; then git checkout --quiet work; else git checkout --quiet -b work FETCH_HEAD; fi
[ -f .halo/submission.yaml ] || { mkdir -p .halo; cp "$package/submission.yaml" .halo/submission.yaml; }
if grep -q '^base_commit: ""' .halo/submission.yaml; then
  sed "s/^base_commit: \"\"/base_commit: \"$base\"/" .halo/submission.yaml > .halo/submission.yaml.new
  mv .halo/submission.yaml.new .halo/submission.yaml
  ok "set base_commit in .halo/submission.yaml to Halo $(printf %.12s "$base")"
fi

step 4 "Install the $APP app"
install="https://github.com/apps/$APP/installations/new/permissions?suggested_target_id=$user_id&repository_ids[]=$fork_id&repository_ids[]=$private_id"
cat <<INSTALL

      GitHub needs one confirmation in the browser. Open this page, check that
      ${green}$login/halo${plain} and ${pink}$login/$name${plain} are the selected repositories, and press ${bold}Install${plain}.

        $install

      If the app is already on your account, add the two repositories there and save.
INSTALL
if command -v xdg-open >/dev/null 2>&1; then xdg-open "$install" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then open "$install" >/dev/null 2>&1 || true; fi
printf '\n      Press Enter when the app is installed. '
read -r _

cat <<DONE

${green}Ready.${plain} You are in ./$dir on the branch "work".

  To submit:
    1. do your work and fill .halo/submission.yaml
    2. git add -A && git commit && git push -u origin work
    3. gh pr create --base main --head work

  The bot answers on the pull request within a minute. Every later push to it is a new submission.
  While a pull request is a draft, the bot ignores it.
DONE
