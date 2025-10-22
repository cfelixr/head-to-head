# Fixes
- Install docker CLI in the container (required by `sam build` sometimes)
- Map Windows creds `${USERPROFILE}/.aws` or `~/.aws`
- Add `sam-version` and `whoami-docker` targets
