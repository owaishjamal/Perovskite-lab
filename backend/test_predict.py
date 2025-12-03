import requests
import json

payload = {
    "bandgap_eV": 1.55,
    "JV_default_Jsc_mAcm2": 23.4,
    "JV_default_Voc_V": 1.10,
    "JV_default_FF": 78.5,
    "Perovskite_deposition_annealing_temperature_C": 100.0,
    "Perovskite_deposition_annealing_time_s": 600.0,
    "Cell_architecture": "n-i-p",
    "Perovskite_composition_short_form": "FA0.8MA0.2PbI3",
    "ETL_material": "TiO2",
    "HTL_material": "Spiro-OMeTAD",
    "Perovskite_deposition_method": "Spin-coating",
    "Additive_type": "MACl",
    "Encapsulation": "Yes"
}

response = requests.post("http://localhost:8000/predict", json=payload)
print("Status:", response.status_code)
print("Response:", json.dumps(response.json(), indent=2))

