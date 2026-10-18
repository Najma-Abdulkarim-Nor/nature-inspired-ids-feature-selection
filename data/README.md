# Datasets

The datasets are **not stored in this repository** because of their size (about 1 GB).
Download them from the Canadian Institute for Cybersecurity and arrange them like this:

```
data/
├── NSL-KDD/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
└── CIC_IDS_2017/
    └── MachineLearningCVE/
        ├── Monday-WorkingHours.pcap_ISCX.csv
        ├── Tuesday-WorkingHours.pcap_ISCX.csv
        ├── Wednesday-workingHours.pcap_ISCX.csv
        ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
        ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
        ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
        ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
        └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

| Dataset | Source | Notes |
|---|---|---|
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html | 41 features + label + difficulty, no header row |
| CIC-IDS-2017 | https://www.unb.ca/cic/datasets/ids-2017.html | Use the `MachineLearningCVE` CSV release |

**Preprocessing done in the notebook**

- NSL-KDD labels are made binary (`normal` → 0, any attack → 1).
- CIC-IDS-2017 per-day CSVs are concatenated, cleaned of infinite or missing values, and
  stratified-downsampled to 150,000 rows (120k train / 30k test) to keep the wrapper search tractable.
