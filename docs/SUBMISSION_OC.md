# OpenClassrooms submission guide

## Zip naming

Global folder: `Realisez_une_analyse_de_sentiments_Demeule_Laureenda`

Individual files (regenerate before upload):
```bash
bash scripts/package_deliverables.sh
```

Files in `livrables/`:
1. `Demeule_Laureenda_1_API_052026.zip` — API + deployment
2. `Demeule_Laureenda_2_scripts_notebook_modelisation_052026.zip`
3. `Demeule_Laureenda_3_dossier_code_052026.zip`
4. `Demeule_Laureenda_4_interface_test_API_052026.zip`
5. `Demeule_Laureenda_5_blog_052026.zip`
6. `Demeule_Laureenda_6_presentation_052026.zip`

## Before submitting

- [ ] Azure API URL works (`/health`, `/predict`) — run `bash scripts/finish_deployment.sh` after `az login`
- [ ] GitHub repo link in README — run `gh auth login` then `bash scripts/setup_github.sh URL && git push`
- [x] Blog updated with final 50k metrics and local screenshots
- [ ] Presentation converted to PDF/PPTX with Azure/GitHub screenshots
- [x] All 6 zips + global zip regenerated (`livrables/Realisez_une_analyse_de_sentiments_Demeule_Laureenda.zip`)

## Platform

Upload on OpenClassrooms project page > Livrables section.
