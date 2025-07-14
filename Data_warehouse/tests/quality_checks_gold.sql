/*
Quality Checks
---------------------------------------------------------------
This script performs quality checks on the 'gold' schema to ensure data integrity, consistency, and accuracy. These checks are :-
1. Uniqueness of surrogate keys in dimension tables
2. Referential integrity between fact and dimension tables
3. Validation of relationships in the data model from analytical purpose
---------------------------------------------------------------
*/
USE DataWarehouse;
GO

SELECT customer_key, COUNT(*) AS duplicate_count
FROM gold.dim_customers
GROUP BY customer_key
HAVING COUNT(*) > 1;

SELECT product_key, COUNT(*) AS duplicate_count
FROM gold.dim_products
GROUP BY product_key
HAVING COUNT(*) > 1;

SELECT *
FROM gold.fact_sales f LEFT JOIN gold.dim_customers c ON c.customer_key = f.customer_key LEFT JOIN gold.dim_products p ON p.product_key = f.product_key
WHERE c.customer_key IS NULL OR f.product_key IS NULL