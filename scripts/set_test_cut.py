import sqlite3

c = sqlite3.connect("data/app.db")
c.execute(
    """
    UPDATE suggestion
    SET action = 'cut_segment', method = 'CUT_BROLL', abc_class = 'B',
        edit_instruction = '删除穿前穿后对比'
    WHERE id = ?
    """,
    ("1ff5663895394523972bf9e5fbdd3d1e",),
)
c.commit()
print("updated")
