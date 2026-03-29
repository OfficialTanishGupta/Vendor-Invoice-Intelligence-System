import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_invoice_data():
    conn = sqlite3.connect('/Users/Tanish_Gupta/OneDrive/Desktop/ML Projects/Invoice Intelligence System/data/inventory.db')

    query = """
    WITH purchase_agg AS (
        SELECT
             p.PONumber,
             COUNT(DISTINCT p.Brand) AS total_brands,
             SUM(p.Quantity) AS total_item-quantity,
             SUM(p.Dollars) AS total_item_dollars,
             AVG(Juliandary(p.ReceivingDate) - julianday(p.PODate)) AS avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )
    SELECT
        vi.PONumber,
        vi.Dollars AS invoice_dollars,
        vi.Freight
        (julianday(vi.InvoiceData)) - julianday(vi.PODate)) AS days_po_to_invoice,
        (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS days_to_pay,
        
    