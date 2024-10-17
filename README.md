# Project Northwind

In today's competitive market, understanding customer behavior and preferences is crucial for the success of businesses. However, the vast amounts of data generated from online transaction processing (OLTP) systems can be overwhelming and challenging to interpret. **Project Northwind** addresses this by employing dimensional reduction and clustering techniques to simplify and group customer data into actionable customer profiles. This empowers businesses with insights that help them make informed, data-driven decisions, ultimately fostering growth and maintaining a competitive edge.

## Features

- **Customer Segmentation with HDBSCAN + UMAP**: Utilizes advanced clustering techniques to model real-world customer segmentation behavior, ensuring effective groupings based on underlying data patterns.
- **SQLite Database Infrastructure**: Powered by the famous Northwind database to simulate a real-world OLTP system.
- **Streamlit User Interface**: Developed a user-friendly UI in Python using Streamlit, connected directly to the SQLite database with optimized SQL queries.
- **Bayesian Optimization with Optuna**: Tuned model hyperparameters using tree-parzen estimators, finding the best model configuration for clustering customer data.
- **Actionable Insights**: Generated detailed customer profiles, which can be leveraged for better decision-making in product development, marketing strategies, and personalized customer experiences.

## Project Highlights

1. **Database Setup**: We utilized the Northwind database in SQLite to simulate a real-world business environment, processing various types of customer, order, and product data.
   
2. **Clustering with HDBSCAN and UMAP**: After preprocessing the data, we applied HDBSCAN (Hierarchical Density-Based Spatial Clustering) and UMAP (Uniform Manifold Approximation and Projection) for clustering and dimensionality reduction. This allowed us to simplify and reveal inherent customer groupings.

3. **Bayesian Optimization**: Using Optuna, we conducted a Bayesian optimization experiment to fine-tune the hyperparameters for our clustering model. This ensured optimal performance and accurate segmentation of customers.

4. **Interactive UI**: Our Streamlit-based app serves as an interface where users can query the SQLite database and visualize the results in an intuitive and accessible manner.

5. **Deliverables**:
   - [Notebook for Clustering & Optimization](#): Detailed analysis and model development documented in this Jupyter Notebook.
   - [PowerPoint Presentation](#): Summary of the clustering insights and their potential business implications.

## Setup Instructions

To get started with **Project Northwind**, follow these steps:

### Prerequisites

- Python 3.7+
- Required Python libraries (listed in `requirements.txt`)
  - Streamlit
  - SQLite3
  - UMAP-learn
  - HDBSCAN
  - Optuna

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/project-northwind.git
   cd project-northwind
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
3. **Run the Streamlit app**:
    ```bash
    streamlit run Dashboard.py
4. **Explore the data**:
    Use the app interface to run SQL queries and view customer segmentation results.

## Usage

Once the Streamlit app is running, users can:

- **Query the Northwind database** for specific customer or order information through the app's intuitive interface.
- **Visualize customer clusters** based on the clustering model, allowing users to understand customer groupings and patterns.
- **Analyze customer profiles** to gain actionable business insights that can inform product development, marketing strategies, and personalized customer experiences.

## Links

- [Clustering & Optimization Notebook](https://github.com/yourusername/project-northwind/notebook-link)
- [PowerPoint Presentation on Results](https://github.com/yourusername/project-northwind/presentation-link)

