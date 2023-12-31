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
import plotly.offline as py
import plotly.graph_objs as go
import country_converter as coco
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)
conn = st.connection('northwind_db', type='sql')
cc = coco.CountryConverter()

def run():
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
      st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

    with tab3:
      st.header("Suppliers")
      st.image("https://static.streamlit.io/examples/owl.jpg", width=200)



if __name__ == "__main__":
    run()
