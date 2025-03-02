# **Sig2Model**  

Sig2Model is a machine learning-powered learned index structure that efficiently maps keys to values while maintaining an adaptive error tolerance. It leverages a combination of **RadixSpline indexing, Gaussian Mixture Models (GMMs), Neural Networks, and Sigmoid-based corrections** to improve data lookup performance.  

---

## **Features**  
- **RadixSpline Learned Index** for fast key-to-position prediction.  
- **GMM-based Data Modeling** to capture key distributions.  
- **Neural Network with Sigmoid Adjustments** to enhance prediction accuracy.  
- **Buffering and Adaptive Retraining** for dynamic updates.  
- **Placeholder Strategy** to handle key insertions efficiently.  

---

## **Core Components**  
### **1. Learned Index (`RadixSpline`)**  
A spline-based index that predicts the approximate position of a queried key.  

### **2. Neural Network (`ComplexNN`)**  
A multi-output neural network that refines index predictions using sigmoid-based corrections.  

### **3. Gaussian Mixture Model (`GMM`)**  
Provides an analytical model of the key distribution to guide retraining and placeholder strategies.  

### **4. Buffering Mechanism (`BufferManager`)**  
Stores recent updates and determines when a full retraining is necessary.  

### **5. Adaptive Retraining (`ControlUnit`)**  
Monitors model accuracy and triggers retraining when errors exceed a threshold.  

### **6. Placeholder Strategy (`PlaceholderStrategy`)**  
Handles dynamic key insertions by introducing artificial placeholders to reduce retraining costs.  

### **7. Sigmoid-Based Correction (`SigmaSigmoid`)**  
Applies fine-tuned sigmoidal transformations to correct lookup errors in real-time.  

---

## **Functionality Overview**  
### **Insertion**  
```cpp
std::vector<double> keys = {1.0, 2.0, 3.0};
std::vector<double> values = {10.0, 20.0, 30.0};
sig2model.insert(keys, values);
```
- Merges new keys while maintaining order.  
- Updates the learned index and refines the neural network model.  

### **Lookup**  
```cpp
std::vector<double> results = sig2model.lookup(2.0);
```
- Predicts position using the learned index.  
- Applies sigmoid-based refinements.  
- Returns a **list of possible values** within the error range.  

### **Updates**  
```cpp
sig2model.update(2.5, 25.0);
```
- Stores updates in **buffer memory** until retraining is triggered.  

### **Retraining**  
- Automatically triggered when the model **exceeds error thresholds**.  
- Uses **GMM distribution analysis** to improve future predictions.  

---

## **Performance Expectations**  
- **Sub-millisecond lookups** for large-scale datasets.  
- **Efficient memory usage** with compressed learned index structures.  
- **Adaptive learning** that improves with additional data.  

---

## **Planned Enhancements**  
- **Support for multi-dimensional indexing.**  
- **Dynamic batch size scaling for retraining.**  
- **Integration with external storage engines.**  

---

## **License**  
This project is licensed under the **MIT License**.  