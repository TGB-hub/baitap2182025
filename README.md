# baitap21082025 nhóm1
!pip -q install osmnx==1.9.3 scikit-fuzzy folium ipywidgets networkx

import osmnx as ox, networkx as nx, numpy as np, folium
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from IPython.display import display, clear_output
import ipywidgets as w

ox.settings.use_cache = True
ox.settings.log_console = False
# Tâm Quận 1 (HCMC), bán kính ~3.5 km cho nhẹ
center = (10.7769, 106.7009)
G = ox.graph_from_point(center, dist=3500, network_type="drive", simplify=True)
G = ox.utils_graph.get_largest_component(G, strongly=False)

# Lấy 20 node ngẫu nhiên (cùng seed để tái lập)
nodes_gdf = ox.graph_to_gdfs(G, edges=False)
nodes_20 = nodes_gdf.sample(20, random_state=42)

# Tạo danh sách lựa chọn cho Dropdown
options = []
for i, (nid, row) in enumerate(nodes_20.iterrows(), 1):
    label = f"#{i}  ({row.y:.5f}, {row.x:.5f})"
    options.append((label, int(nid)))

start_dd = w.Dropdown(options=options, description="Start:")
end_dd   = w.Dropdown(options=options, description="End:")

# Các yếu tố ảnh hưởng giá
peak_dd    = w.Dropdown(options=[("Off-peak", 0.0), ("Moderate", 0.5), ("Peak", 1.0)],
                        value=0.0, description="Peak hour")
rain_dd    = w.Dropdown(options=[("Clear", 0.0), ("Light rain", 0.5), ("Heavy rain", 1.0)],
                        value=0.0, description="Weather")
voucher_dd = w.Dropdown(options=[("None", 0.0), ("Small", 0.5), ("Large", 1.0)],
                        value=0.0, description="Voucher")

btn = w.Button(description="Compute & Show", button_style="primary")
out = w.Output()

display(w.VBox([w.HBox([start_dd, end_dd]),
                w.HBox([peak_dd, rain_dd, voucher_dd]),
                btn, out]))
# Antecedents
distance = ctrl.Antecedent(np.arange(0, 21, 0.1), 'distance')  # km (0–20+)
peak     = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'peak')    # 0 off → 1 peak
rain     = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'rain')    # 0 clear → 1 heavy
voucher  = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'voucher')  # 0 none → 1 large

# Consequent: hệ số nhân giá (0.6–1.8)
mult     = ctrl.Consequent(np.arange(0.6, 1.81, 0.01), 'mult')

# Memberships
distance['short']  = fuzz.trapmf(distance.universe, [0, 0, 2, 5])
distance['medium'] = fuzz.trimf(distance.universe, [3, 8, 13])
distance['long']   = fuzz.trapmf(distance.universe, [10, 14, 20, 20])

peak['low']    = fuzz.trapmf(peak.universe, [0, 0, 0.2, 0.4])
peak['medium'] = fuzz.trimf(peak.universe, [0.3, 0.5, 0.7])
peak['high']   = fuzz.trapmf(peak.universe, [0.6, 0.8, 1, 1])

rain['clear']  = fuzz.trapmf(rain.universe, [0, 0, 0.2, 0.4])
rain['light']  = fuzz.trimf(rain.universe, [0.3, 0.5, 0.7])
rain['heavy']  = fuzz.trapmf(rain.universe, [0.6, 0.8, 1, 1])

voucher['none']  = fuzz.trapmf(voucher.universe, [0, 0, 0.15, 0.3])
voucher['small'] = fuzz.trimf(voucher.universe, [0.2, 0.5, 0.8])
voucher['large'] = fuzz.trapmf(voucher.universe, [0.7, 0.85, 1, 1])

mult['low']    = fuzz.trapmf(mult.universe, [0.6, 0.65, 0.8, 0.95])
mult['normal'] = fuzz.trimf(mult.universe, [0.9, 1.0, 1.1])
mult['high']   = fuzz.trapmf(mult.universe, [1.1, 1.3, 1.8, 1.8])

# Rules (cô đọng nhưng bao trùm)
rules = [
    ctrl.Rule(peak['high'] | rain['heavy'], mult['high']),
    ctrl.Rule(distance['long'] & (peak['medium'] | peak['high']), mult['high']),
    ctrl.Rule(distance['short'] & peak['low'] & rain['clear'], mult['low']),
    ctrl.Rule(peak['medium'] & rain['light'], mult['normal']),
    ctrl.Rule(distance['medium'] & peak['low'] & rain['clear'], mult['normal']),
    ctrl.Rule(voucher['large'], mult['low']),
    ctrl.Rule(voucher['small'] & (peak['high'] | rain['heavy']), mult['normal']),
    ctrl.Rule((peak['low'] & rain['clear']) & voucher['none'], mult['normal']),
]

mult_ctrl = ctrl.ControlSystem(rules)
# Tham số tính giá gốc (có thể chỉnh theo nhu cầu)
BASE_FARE = 12000      # VND mở cửa
PER_KM    = 9000       # VND/km
CURRENCY  = "VND"

def compute_price_k_vnd(dist_km, peak_val, rain_val, voucher_val):
    sim = ctrl.ControlSystemSimulation(mult_ctrl)
    sim.input['distance'] = float(dist_km)
    sim.input['peak']     = float(peak_val)
    sim.input['rain']     = float(rain_val)
    sim.input['voucher']  = float(voucher_val)
    sim.compute()
    m = sim.output['mult']
    raw = BASE_FARE + PER_KM * dist_km
    price = raw * m
    return m, raw, price

def to_coords(G, route):
    return [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]

def on_click(_):
    with out:
        clear_output()
        s = start_dd.value
        e = end_dd.value
        if s == e:
            print(" Start và End phải khác nhau.")
            return
        # Shortest path by length
        route = nx.shortest_path(G, s, e, weight='length')
        meters = sum(ox.utils_graph.get_route_edge_attributes(G, route, 'length'))
        km = round(meters / 1000.0, 3)

        # Fuzzy price
        m, raw, price = compute_price_k_vnd(km, peak_dd.value, rain_dd.value, voucher_dd.value)

        # Map
        mid = center
        mapp = folium.Map(location=mid, zoom_start=13, tiles="cartodbpositron")
        coords = to_coords(G, route)
        folium.PolyLine(coords, weight=6, color="green", opacity=0.8).add_to(mapp)
        folium.Marker(coords[0], tooltip="START", icon=folium.Icon(color="blue")).add_to(mapp)
        folium.Marker(coords[-1], tooltip="END", icon=folium.Icon(color="red")).add_to(mapp)

        # Show summary
        print(f" Distance: {km:.3f} km")
        print(f" Peak: {peak_dd.label}    Weather: {rain_dd.label}    Voucher: {voucher_dd.label}")
        print(f"  Fuzzy multiplier m = {m:.3f}")
        print(f" Raw fare = {raw:,.0f} {CURRENCY}")
        print(f" Final fare ≈ {price:,.0f} {CURRENCY}")

        display(mapp)

btn.on_click(on_click)
print("Chọn Start/End và các điều kiện rồi bấm *Compute & Show*.")
