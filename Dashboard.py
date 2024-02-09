# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import pandas
import plotly.express as px
import plotly.graph_objs as go
import country_converter as coco
from streamlit.logger import get_logger
from utils import run_umap

LOGGER = get_logger(__name__)
conn = st.connection('northwind_db', type='sql')
cc = coco.CountryConverter()

def build_dash_board() -> None:
    tab1, tab2, tab3 = st.tabs(["Sales", "Customers", "Suppliers"])
    with tab1:
      st.header("Sales")
      st.write("### Monthly Sales Trend")
      #returns a dataframe
      monthly_sales = conn.query('SELECT strftime(\'%m\',OrderDate) AS Month, SUM(Quantity * UnitPrice) AS MonthlySales FROM Orders INNER JOIN [Order Details] ON Orders.OrderID == [Order Details].OrderID GROUP BY Month;')
      #st.dataframe(monthly_sales)
      st.line_chart(monthly_sales, x = "Month", y = "MonthlySales")

      st.write("### Sales by Category")
      category_sales = conn.query('SELECT Categories.CategoryName AS Category, SUM(Quantity * [Order Details].UnitPrice) AS SalesByCategory FROM Orders INNER JOIN [Order Details] ON Orders.OrderID == [Order Details].OrderID INNER JOIN Products ON [Order Details].ProductID == Products.ProductID INNER JOIN Categories ON Products.CategoryID == Categories.CategoryID GROUP BY Categories.CategoryName;')
      #st.dataframe(category_sales)
      st.bar_chart(category_sales, x = "Category", y = "SalesByCategory")

      st.write("### Sales by Region")
      region_sales = conn.query('SELECT ShipCountry, SUM(Quantity * UnitPrice) AS SalesByRegion FROM Orders INNER JOIN [Order Details] ON Orders.OrderID == [Order Details].OrderID GROUP BY ShipRegion;')
      st.dataframe(region_sales)
      
      country_codes = cc.pandas_convert(series=region_sales.ShipCountry, to='ISO3')  
      z = region_sales.SalesByRegion

      layout = dict(geo={'scope': "world"})
      data = dict(
          type='choropleth',
          locations=country_codes,
          locationmode='ISO-3',
          colorscale='Viridis',
          z=z)
      map = go.Figure(data=[data], layout=layout)
      map.update_geos(projection_type="orthographic")
      map.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0})
      st.plotly_chart(map)

      st.write("### Top 10 Sales")
      top_10_sales = conn.query('WITH SalesRanked AS (SELECT Products.ProductName AS ProductName, SUM(Quantity * [Order Details].UnitPrice) AS Sales FROM Orders INNER JOIN [Order Details] ON Orders.OrderID == [Order Details].OrderID INNER JOIN Products ON [Order Details].ProductID == Products.ProductID GROUP BY [Order Details].ProductID ORDER BY Sales DESC) SELECT * FROM SalesRanked LIMIT 10;')
      #st.dataframe(top_10_sales)
      st.bar_chart(top_10_sales, x = "ProductName", y = "Sales")

      st.write("### Sales by Employee")
      employee_sales = conn.query('SELECT Employees.FirstName || " " || Employees.LastName AS EmployeeName, SUM(Quantity * [Order Details].UnitPrice) AS SalesByEmployee FROM Employees INNER JOIN Orders ON Employees.EmployeeID == Orders.EmployeeID INNER JOIN [Order Details] ON Orders.OrderID == [Order Details].OrderID GROUP BY EmployeeName;')
      #st.dataframe(employee_sales)
      st.bar_chart(employee_sales, x = "EmployeeName", y = "SalesByEmployee")

    with tab2:
      st.header("Customers")
      # Customer Segmentation: A pie chart showing segmentation of customers based on their purchase behavior
      # TODO: use UMAP for dimensional reduction and HDBSCAN for clustering
      st.write("### Customer Segmentation")
      customer_data = conn.query('SELECT Orders.OrderID, Orders.OrderDate, Orders.ShippedDate, Customers.Country AS CustomerCountry, Customers.City AS CustomerCity, Customers.Region AS CustomerRegion, Products.ProductID, Products.ProductName, Products.CategoryID, [Order Details].UnitPrice, [Order Details].Quantity, [Order Details].Discount, Categories.CategoryName, Suppliers.Country AS SupplierCountry, Suppliers.Region AS SupplierRegion, ([Order Details].UnitPrice * [Order Details].Quantity) - ([Order Details].UnitPrice * [Order Details].Quantity * [Order Details].Discount) AS TotalPrice FROM Orders INNER JOIN Customers ON Orders.CustomerID = Customers.CustomerID INNER JOIN [Order Details] ON Orders.OrderID = [Order Details].OrderID INNER JOIN Products ON [Order Details].ProductID = Products.ProductID INNER JOIN Categories ON Products.CategoryID = Categories.CategoryID INNER JOIN Suppliers ON Products.SupplierID = Suppliers.SupplierID WHERE Orders.OrderDate BETWEEN \'2010-01-01\' AND \'2020-12-31\' ORDER BY Orders.OrderDate;')
      st.dataframe(customer_data)

      # transform categorical data
      df = pandas.DataFrame(customer_data).dropna()
      # columns: OrderID, OrderDate, ShippedDate, CustomerCountry, CustomerCity, CustomerRegion, ProductID, ProductName, CategoryID, UnitPrice, Quantity, Discount, CategoryName, SupplierCountry,SupplierRegion,TotalPrice
      cols_to_transform = ['OrderDate', 'ShippedDate', 'CustomerCountry', 'CustomerCity', 'CustomerRegion', 'ProductName', 'CategoryName', 'SupplierCountry', 'SupplierRegion']
      cols_to_retain = ['OrderID', 'TotalPrice', 'UnitPrice', 'Quantity', 'ProductID', 'CategoryID', 'Discount']
     
      embedding = run_umap(df, cols_to_transform, cols_to_retain)
      
      fig = px.scatter_3d(data_frame = df, x = embedding[:, 0], y = embedding[:, 1], z = embedding[:, 2], size_max=10, color='CustomerRegion')
      st.plotly_chart(fig)
      

      # Customer Lifetime Value (CLV): A bar chart showing the CLV of different customer segments
      st.write("### Customer Lifetime Value")

      #Top Customers: A list or bar chart of top customers by sales
      st.write("### Top Customers")
      top_customers = conn.query('SELECT Customers.CustomerID AS CustomerName, SUM(Quantity * UnitPrice) AS TotalOrder FROM Customers INNER JOIN Orders ON Customers.CustomerID = Orders.CustomerID INNER JOIN [Order Details] ON Orders.OrderID = [Order Details].OrderID GROUP BY CustomerName ORDER BY TotalOrder DESC LIMIT 10')
      st.bar_chart(top_customers, x = "CustomerName", y = "TotalOrder")

      #Customer Geographic Distribution: A map showing where customers are located, which can be filtered by the date range and category.
      st.write("### Customer Geographic Distribution")
      geographic_distribution = conn.query('SELECT Country, COUNT(*) AS NumCustomers FROM Customers GROUP BY Country')
      st.bar_chart(geographic_distribution, x = "Country", y = "NumCustomers")

      country_codes = cc.pandas_convert(series=geographic_distribution.Country, to='ISO3')  
      z = geographic_distribution.NumCustomers

      layout = dict(geo={'scope': "world"})
      data = dict(
          type='choropleth',
          locations=country_codes,
          locationmode='ISO-3',
          colorscale='Viridis',
          z=z)
      map = go.Figure(data=[data], layout=layout)
      map.update_geos(projection_type="orthographic")
      map.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0})
      st.plotly_chart(map)

      #Orders by Customer: A bar chart showing the number of orders by customer
      st.write("Orders by Customer")
      orders_by_customer = conn.query('SELECT Customers.CustomerID AS CustomerName, COUNT(*) AS TotalOrder FROM Customers INNER JOIN Orders ON Customers.CustomerID = Orders.CustomerID GROUP BY Customers.CustomerID ORDER BY TotalOrder DESC')
      st.bar_chart(orders_by_customer, x="CustomerName", y="TotalOrder")


    with tab3:
      st.header("Suppliers")

      #Supplier Performance: Metrics on delivery time, completeness, and correctness from different suppliers
      st.write("### Supplier Performance")

      #Supplier Geographic Distribution: A map showing where suppliers are located
      st.write("### Supplier Geographic Distribution")
      supplier_distribution = conn.query("SELECT Country, COUNT(*) AS NumSuppliers FROM Suppliers GROUP BY Country")
      st.bar_chart(supplier_distribution, x="Country", y="NumSuppliers")
      country_codes = cc.pandas_convert(series=supplier_distribution.Country, to='ISO3')  
      z = supplier_distribution.NumSuppliers

      layout = dict(geo={'scope': "world"})
      data = dict(
          type='choropleth',
          locations=country_codes,
          locationmode='ISO-3',
          colorscale='Viridis',
          z=z)
      map = go.Figure(data=[data], layout=layout)
      map.update_geos(projection_type="orthographic")
      map.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0})
      st.plotly_chart(map)

      #Products by Supplier: A bar chart showing the number of products supplied by each supplier
      st.write("Products by Supplier")
      products_by_supplier = conn.query("SELECT CompanyName, COUNT(*) AS NumProducts FROM Products INNER JOIN Suppliers ON Products.SupplierID = Suppliers.SupplierID GROUP BY Suppliers.SupplierID")
      st.bar_chart(products_by_supplier, x="CompanyName", y="NumProducts")

      #Inventory Levels by Supplier: A bar chart showing current inventory levels of different products by supplier
      st.write("Inventory Levels by Supplier")
      inventory_levels = conn.query("SELECT CompanyName, SUM(UnitsInStock) AS Inventory FROM Products INNER JOIN Suppliers ON Products.SupplierID = Suppliers.SupplierID GROUP BY Suppliers.SupplierID")
      st.bar_chart(inventory_levels, x="CompanyName", y="Inventory")

      #Orders by Supplier: A bar chart showing the number of orders by supplier
      st.write("Orders by Supplier")
      orders_by_suppliers = conn.query("SELECT CompanyName, COUNT(DISTINCT(OrderID)) AS NumOrders FROM [Order Details] INNER JOIN Products ON [Order Details].ProductID = Products.ProductID INNER JOIN Suppliers ON Products.SupplierID = Suppliers.SupplierID GROUP BY Suppliers.SupplierID")
      st.bar_chart(orders_by_suppliers, x="CompanyName", y="NumOrders")




if __name__ == "__main__":
    #st.set_page_config(page_title="Dashboard Demo - Northwind Database", page_icon="📹")
    st.markdown("# Dashboard Demo - Northwind Database")
    st.sidebar.header("Dashboard Demo")
    st.write(
        """The dashboard looks at Northwind database and displays the data in a variety of plots. The dashboard can be filtered by your selections on the left. Potential users for this dashboard are businesses trying to analyze their sales for improvements and new investment areas."""
    )
    build_dash_board()
