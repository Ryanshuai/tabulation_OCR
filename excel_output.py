import pandas as pd
import numpy as np

list_l = [[11, 12, 13, 14, 15], [21, 22, 23, 24, 25], [31, 32, 33, 34, 35]]
date_range = pd.date_range(start="20180701", periods=3)
df = pd.DataFrame(list_l, index=date_range, columns=['a', 'b', 'c', 'd', 'e'])
print(df)


df.to_excel('excel_output.xls')

# xl = pd.ExcelFile("master_transcript_Chinese_Ren.xlsx")