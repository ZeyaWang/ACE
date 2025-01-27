# Data Dictionary

This document describes the structure and contents of datasets stored in the Google drive.

## **1. Pickle File (`data.pkl`)**
Pickle files store Python objects in a serialized format.

**Structure:**
- **Data Type:** Dictionary
- **Keys and Descriptions:**
  | Key        | Data Type | Description |
  |-----------|----------|-------------|
  | `X_train`  | ndarray  | Training feature set |
  | `X_test`   | ndarray  | Test feature set |
  | `y_train`  | ndarray  | Training labels |
  | `y_test`   | ndarray  | Test labels |
  | `metadata` | dict     | Additional information (e.g., feature names, dataset source) |

---

## **2. HDF5 File (`data.h5`)**
HDF5 files store hierarchical data structures and are commonly used for large datasets.

**Structure:**
- **Groups and Datasets:**
  | Group / Dataset | Data Type | Description |
  |---------------|----------|-------------|
  | `/train/X`    | ndarray  | Training feature set |
  | `/train/y`    | ndarray  | Training labels |
  | `/test/X`     | ndarray  | Test feature set |
  | `/test/y`     | ndarray  | Test labels |
  | `/metadata`   | dict     | Additional metadata (e.g., feature names) |

---

## **3. RDA File (`data.rda`)**
RDA files store R objects, such as data frames and lists.

**Structure:**
- **Objects within the file:**
  | Object Name  | Data Type  | Description |
  |-------------|-----------|-------------|
  | `df_train`  | DataFrame | Training dataset (features + labels) |
  | `df_test`   | DataFrame | Test dataset (features + labels) |
  | `params`    | List      | Hyperparameters used in modeling |
  | `metadata`  | List      | Dataset description and column names |

---

## **4. Additional Pickle File (`model.pkl`)**
This file stores a trained machine learning model.

**Structure:**
- **Data Type:** Serialized Machine Learning Model
- 
