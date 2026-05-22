# OpenClassrooms submission guide

## Zip naming

Global folder: `Realisez_une_analyse_de_sentiments_Demeule_Laureenda`

Regenerate before upload:

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

- [x] Azure API URL works — https://air-paradis-sentiment-P7.azurewebsites.net/health
- [x] GitHub repo link in README — https://github.com/molly-muffin/P7_OC_AirParadis
- [x] Blog updated with final 50k metrics and screenshots
- [x] Presentation PDF/PPTX with Azure/GitHub screenshots (`docs/presentation.pdf`, `docs/presentation.pptx`)
- [x] All 6 zips + global zip regenerated

## Upload on OpenClassrooms

1. Open your OC project page → section **Livrables**
2. Upload `livrables/Realisez_une_analyse_de_sentiments_Demeule_Laureenda.zip`
   (or the 6 individual zips if the platform requires them separately)
3. In the project text field, paste:
   - **API** : https://air-paradis-sentiment-P7.azurewebsites.net
   - **GitHub** : https://github.com/molly-muffin/P7_OC_AirParadis
4. Submit for evaluation

## Verify after upload

```bash
curl -s https://air-paradis-sentiment-P7.azurewebsites.net/health | python -m json.tool
pytest tests/ -q
```
