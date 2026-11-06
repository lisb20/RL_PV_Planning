BASE = 1e8

def facilityid_2_graphid(faci_id, N_500_0 = 9, N_500_1 = 16, N_220 = 100, N_110 = 803):
    if faci_id // BASE == 1:
        g_id = faci_id % BASE + N_110
    elif faci_id // BASE == 2:
        g_id = N_500_0 + faci_id % BASE + N_110
    elif faci_id // BASE == 3:
        g_id = N_500_0 + N_500_1 + faci_id % BASE + N_110
    elif faci_id // BASE == 4:
        g_id = faci_id % BASE

    return g_id
