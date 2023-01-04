---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="view-in-github" colab_type="text" -->
<a href="https://colab.research.google.com/github/EhsanAghazadeh/Metaphors_in_PLMs/blob/main/Metaphor_Demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

```python id="H16lCkpJCcnJ"
! pip install datasets==1.18.3
! pip install transformers==4.18.0
! rm -r Metaphors_in_PLMs/
! git clone https://github.com/EhsanAghazadeh/Metaphors_in_PLMs
! git clone https://github.com/mohsenfayyaz/edge-probing-datasets.git
```

```python colab={"base_uri": "https://localhost:8080/"} id="1fvPbruBCpHK" outputId="f36e58c6-4522-4642-f2e1-eccea46c96a7"
%env MDL_CODE=/content/Metaphors_in_PLMs/source_code/scripts/mdl_probing.py
%env EDGE_CODE=/content/Metaphors_in_PLMs/source_code/scripts/edge_probing.py
```

<!-- #region id="DfNqxes7UEVi" -->
# Edge Probing
<!-- #endregion -->

```python id="g041niRlFeX7"
! python3 $EDGE_CODE bert-base-uncased trofi 0
```

<!-- #region id="NUH6T1C3T3Iq" -->
Results are saved in /content/edge_probing_results
<!-- #endregion -->

<!-- #region id="Tq2A_tE7UIcm" -->
# MDL Probing
<!-- #endregion -->

```python id="JSx2flU7Fiyz"
! python3 $MDL_CODE bert-base-uncased trofi 0
```

<!-- #region id="zFEPTwFLU5rD" -->
Results are saved in /content/mdl_results
<!-- #endregion -->

```python id="lowM992vVDha"

```
