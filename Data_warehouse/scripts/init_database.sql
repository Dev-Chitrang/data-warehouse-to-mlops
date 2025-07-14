/*
Create Database and Schemas
---------------------------------------------------------------
This script creates a new database named 'DataWarehouse' after checking if it already exists.
If exists, it drops and recreates the database.
Additionally, the script sets up the 'bronze', 'silver', and 'gold' schemas within the 'DataWarehouse' database.
---------------------------------------------------------------
*/


USE master;
GO

IF EXISTS (SELECT 1
FROM sys.databases
WHERE name = 'DataWarehouse')
BEGIN
    -- ALTER DATABASE DataWarehouse SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
    DROP DATABASE DataWarehouse;
    PRINT 'Database DataWarehouse dropped';
END;
GO

CREATE DATABASE DataWarehouse;
GO

USE DataWarehouse;
GO

CREATE SCHEMA bronze;
GO

CREATE SCHEMA silver;
GO

CREATE SCHEMA gold;
GO