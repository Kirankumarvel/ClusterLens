# ClusterLens 🔍
*A Visual Exploration of Dimensionality Reduction with t-SNE, UMAP, and PCA*

## 📊 Overview

ClusterLens is a Python project that demonstrates how high-dimensional data (3D in this case) can be simplified into 2D for easier visualization and analysis using popular dimensionality reduction techniques:  
- 🌀 **t-SNE (t-distributed Stochastic Neighbor Embedding)**  
- 🌌 **UMAP (Uniform Manifold Approximation and Projection)**  
- 📐 **PCA (Principal Component Analysis)**

The project uses synthetic data with four clear clusters and compares how each method preserves or distorts the original structure in 2D.

---

## 📦 What’s Inside

- 📁 **Synthetic Data Generator** (using `make_blobs`)
- 📈 **3D Plot with Plotly** for initial data visualization
- 🧽 **Data Standardization** using `StandardScaler`
- 🔽 **2D projections** using:
  - t-SNE  
  - UMAP  
  - PCA
- 🖼️ **Beautiful Matplotlib & Plotly visualizations** for side-by-side comparisons

---

## 🧑‍🏫 Designed for Learning

This project is made simple and clear, ideal for:
- ML beginners exploring dimensionality reduction
- Data scientists comparing methods
- Educators and students learning by visualization

---

## 🚀 How to Run

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/clusterlens.git
   cd clusterlens
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```


## 📦 Dependencies
- numpy
- pandas
- matplotlib
- scikit-learn
- umap-learn
- plotly

Install via:
```bash
pip install numpy pandas matplotlib scikit-learn umap-learn plotly
```

## 📸 Sample Output
| t-SNE | UMAP | PCA |
|-------|------|-----|

## 🤔 Why It Matters
Dimensionality reduction is crucial in:
- Visualizing complex data
- Speeding up ML algorithms
- Discovering hidden structures

This project helps visualize how each method treats data differently.

## 🧒 Bonus: Kid-Friendly Explanation
Imagine you have lots of colorful marbles in a 3D box (like X, Y, Z directions). But it's hard to see how they group together! So, we squish that box into a flat picture using magic lenses — t-SNE, UMAP, and PCA — and see which lens shows the marbles grouped the best! 🪄🎨

## 📄 License
MIT License

## 💡 Inspired by
- Real-world applications in AI and bioinformatics
- scikit-learn documentation
- UMAP & t-SNE visualizations in research papers

## 🙌 Contribute
Feel free to fork, improve, and make it your own! PRs are welcome.
