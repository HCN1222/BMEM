import matplotlib
matplotlib.use('TkAgg') 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import os
import sys

print("1. Starting script...")

# ==========================================
# 1. Load Data & Cache Setup
# ==========================================
try:
    df = pd.read_parquet('./data/preprocessed_data/final_vectors_train.parquet')
    hmm_params = np.load('./outputs/result_20260315_174940/trained_hmm_params.npz')
    states = hmm_params['decoded_states']
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit()

if len(df) != len(states):
    min_len = min(len(df), len(states))
    df = df.iloc[-min_len:].reset_index(drop=True)
    states = states[-min_len:]

df['State'] = states

unique_states = np.unique(states)
cmap = plt.get_cmap('tab10')
state_colors = {state: cmap(i % 10) for i, state in enumerate(unique_states)}

kline_cache = {}
def get_kline_data(stock_id, date_str):
    if stock_id not in kline_cache:
        file_path = f"./data/stocks/{stock_id}_2021-06-30_to_2026-02-11.parquet"
        if os.path.exists(file_path):
            try:
                kdf = pd.read_parquet(file_path)
                kdf['date'] = kdf['date'].astype(str).str[:10]
                kdf.set_index('date', inplace=True)
                kline_cache[stock_id] = kdf
            except:
                kline_cache[stock_id] = None
        else:
            kline_cache[stock_id] = None
            
    kdf = kline_cache[stock_id]
    if kdf is not None and date_str in kdf.index:
        row = kdf.loc[date_str]
        if isinstance(row, pd.DataFrame): row = row.iloc[0]
        return row
    assert 0
    return None

# ==========================================
# 2. Window Settings & Global Variables
# ==========================================
WINDOW_SIZE = 60
current_idx = 0 

is_dragging = False
press_x = None
press_idx = None

fig, (ax0, ax1, ax2, ax3) = plt.subplots(
    nrows=4, ncols=1, figsize=(16, 12), sharex=True,
    gridspec_kw={'height_ratios': [3, 2, 2, 0.5]} 
)

plt.subplots_adjust(bottom=0.15, hspace=0.3) 

hover_vlines = []
hover_hlines = []
tooltip = fig.text(0, 0, "", va="bottom", ha="left", 
                   bbox=dict(boxstyle="round,pad=0.4", fc="#ffffe0", ec="gray", alpha=0.95),
                   fontsize=10, zorder=100, visible=False)

# ==========================================
# 3. Drawing Function
# ==========================================
def draw_chart():
    global hover_vlines, hover_hlines
    
    ax0.clear()
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    sub_df = df.iloc[current_idx : current_idx + WINDOW_SIZE].copy()
    if sub_df.empty: return
        
    x_positions = np.arange(len(sub_df)) 
    date_labels = sub_df['date'].astype(str).str[:10].tolist()
    
    # ----- 區塊 1: K-Line (Stock Price) -----
    valid_kline_drawn = False
    
    # 用來記錄當前區間的最高價與最低價，以便縮放 Y 軸
    window_min_price = float('inf')
    window_max_price = float('-inf')
    
    for i, (idx, row) in enumerate(sub_df.iterrows()):
        stock_id = str(row['stock_id'])
        date_str = str(row['date'])[:10]
        k_data = get_kline_data(stock_id, date_str)
        if k_data is not None:
            valid_kline_drawn = True
            o, h, l, c = k_data['open'], k_data['max'], k_data['min'], k_data['close']
            color = 'red' if c >= o else 'green'
            
            # 更新 Y 軸縮放範圍
            if l < window_min_price: window_min_price = l
            if h > window_max_price: window_max_price = h
            
            ax0.vlines(x=i, ymin=l, ymax=h, color=color, linewidth=1.5)
            body_bottom, body_height = min(o, c), max(o, c) - min(o, c)
            if body_height == 0: body_height = (h - l) * 0.001 if h > l else 0.1
            ax0.bar(x=i, height=body_height, bottom=body_bottom, width=0.5, color=color)

    ax0.set_title(f'Stock Price (K-Line) | Index: {current_idx} to {current_idx + len(sub_df) - 1}', fontsize=14)
    ax0.grid(True, linestyle='--', alpha=0.5)
    
    if valid_kline_drawn and window_min_price != float('inf'):
        # 設定 Y 軸上下限，並加上 5% 的留白，才不會讓 K 線頂到天花板或地板
        padding = (window_max_price - window_min_price) * 0.05
        if padding == 0: padding = window_max_price * 0.01 # 避免高低價相同時出錯
        ax0.set_ylim(window_min_price - padding, window_max_price + padding)
    else:
        ax0.text(0.5, 0.5, "No K-Line Data Found", ha='center', va='center', transform=ax0.transAxes)

    # ----- 區塊 2: net_buy (Actual Amount) -----
    if 'net_buy' in sub_df.columns:
        net_buy_colors = ['#ff6666' if val > 0 else '#66ff66' for val in sub_df['net_buy']]
        ax1.bar(x_positions, sub_df['net_buy'], color=net_buy_colors, width=0.6)
    
    ax1.set_title('Actual Net Buy/Sell (net_buy)', fontsize=12)
    ax1.axhline(0, color='black', linewidth=0.8) 
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ----- 區塊 3: z_t (Feature) -----
    zt_colors = ['red' if val > 0 else 'green' for val in sub_df['z_t']]
    ax2.bar(x_positions, sub_df['z_t'], color=zt_colors, width=0.6)
    
    if WINDOW_SIZE <= 100:
        for i, (idx, row) in enumerate(sub_df.iterrows()):
            val = row['z_t']
            stock_id = str(row['stock_id'])
            y_pos = val + (abs(val) * 0.05) if val > 0 else val - (abs(val) * 0.05)
            va_align = 'bottom' if val > 0 else 'top'
            ax2.text(i, y_pos, stock_id, ha='center', va=va_align, fontsize=8, rotation=90, color='black')

    ax2.set_title('Feature (z_t)', fontsize=12)
    ax2.axhline(0, color='black', linewidth=0.8) 
    ax2.grid(True, linestyle='--', alpha=0.5)

    # ----- 區塊 4: Predicted States -----
    c_list = [state_colors[s] for s in sub_df['State']]
    ax3.bar(x_positions, [1]*len(sub_df), color=c_list, width=1.0)
    
    ax3.set_title('Predicted States', fontsize=12)
    ax3.set_yticks([]) 
    ax3.set_ylabel('State', rotation=0, labelpad=20, va='center')
    ax3.set_xlim(-0.5, len(sub_df) - 0.5) 

    ax3.set_xticks(x_positions)
    step = max(1, WINDOW_SIZE // 30)
    ax3.set_xticks(x_positions[::step])
    ax3.set_xticklabels(date_labels[::step], rotation=45, ha='right', fontsize=9)
    
    # 建立十字線
    hover_vlines = [ax.axvline(0, color='#333333', lw=1, ls='--', visible=False, zorder=99) for ax in (ax0, ax1, ax2, ax3)]
    hover_hlines = [ax.axhline(0, color='#333333', lw=1, ls='--', visible=False, zorder=99) for ax in (ax0, ax1, ax2, ax3)]

    fig.canvas.draw_idle()

# ==========================================
# 4. Mouse Interactive Events
# ==========================================
def on_scroll(event):
    global current_idx, WINDOW_SIZE
    if event.inaxes is None or event.xdata is None: return

    scale_factor = 1.3 
    if event.button == 'up': new_window = int(WINDOW_SIZE / scale_factor)
    elif event.button == 'down': new_window = int(WINDOW_SIZE * scale_factor)
    else: return

    new_window = max(10, min(new_window, 400))
    if new_window == WINDOW_SIZE: return

    relative_x = event.xdata / WINDOW_SIZE
    current_idx = current_idx + int(event.xdata) - int(relative_x * new_window)
    WINDOW_SIZE = new_window
    current_idx = max(0, min(current_idx, len(df) - WINDOW_SIZE))
    draw_chart()

def on_press(event):
    global is_dragging, press_x, press_idx
    if event.button == 1 and event.inaxes is not None:
        is_dragging = True
        press_x = event.x 
        press_idx = current_idx

def on_motion(event):
    global current_idx
    if is_dragging and event.inaxes is not None:
        dx_pixels = press_x - event.x
        shift = int((dx_pixels / event.inaxes.bbox.width) * WINDOW_SIZE)
        if abs(shift) >= 1:
            new_idx = press_idx + shift
            new_idx = max(0, min(new_idx, len(df) - WINDOW_SIZE))
            if new_idx != current_idx:
                current_idx = new_idx
                draw_chart()
        return

    if event.inaxes in (ax0, ax1, ax2, ax3):
        # 1. 顯示垂直線 (貫穿)
        for vline in hover_vlines:
            vline.set_xdata([event.xdata, event.xdata])
            vline.set_visible(True)
            
        # 2. 顯示水平線 (單一區塊)
        for i, ax in enumerate((ax0, ax1, ax2, ax3)):
            if event.inaxes == ax:
                hover_hlines[i].set_ydata([event.ydata, event.ydata])
                hover_hlines[i].set_visible(True)
            else:
                hover_hlines[i].set_visible(False)
                
        # 3. 提示框
        x_idx = int(round(event.xdata))
        sub_df = df.iloc[current_idx : current_idx + WINDOW_SIZE]
        if 0 <= x_idx < len(sub_df):
            row = sub_df.iloc[x_idx]
            date_str = str(row['date'])[:10]
            stock_id = str(row['stock_id'])
            val_y = event.ydata
            
            y_label = ""
            if event.inaxes == ax0: y_label = f"Price: {val_y:.2f}"
            elif event.inaxes == ax1: y_label = f"Net Buy: {val_y:,.0f}"
            elif event.inaxes == ax2: y_label = f"z_t: {val_y:.4f}"
            elif event.inaxes == ax3: y_label = f"State Color Band"
            
            text_str = f"Date: {date_str}\nStock: {stock_id}\n{y_label}"
            
            fig_w, fig_h = fig.canvas.get_width_height()
            tx = (event.x + 15) / fig_w
            ty = (event.y + 15) / fig_h
            if tx > 0.85: tx = (event.x - 120) / fig_w 
            if ty > 0.85: ty = (event.y - 60) / fig_h  
            
            tooltip.set_text(text_str)
            tooltip.set_position((tx, ty))
            tooltip.set_visible(True)
        else:
            tooltip.set_visible(False)
            
        fig.canvas.draw_idle()
    else:
        if tooltip.get_visible():
            for vline in hover_vlines: vline.set_visible(False)
            for hline in hover_hlines: hline.set_visible(False)
            tooltip.set_visible(False)
            fig.canvas.draw_idle()

def on_release(event):
    global is_dragging
    is_dragging = False

fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# ==========================================
# 5. Buttons & Search Behaviors
# ==========================================
def go_next(event):
    global current_idx
    if current_idx + WINDOW_SIZE < len(df):
        current_idx += WINDOW_SIZE
        draw_chart()

def go_prev(event):
    global current_idx
    current_idx = max(0, current_idx - WINDOW_SIZE)
    draw_chart()

def search_stock(text):
    global current_idx
    matches = df[df['stock_id'].astype(str) == text].index
    if len(matches) > 0:
        current_idx = matches[0]
        draw_chart()
        print(f"Jumped to stock_id: {text}")
    else:
        print(f"Stock_id not found: {text}")

ax_prev = plt.axes([0.65, 0.03, 0.1, 0.05])
ax_next = plt.axes([0.76, 0.03, 0.1, 0.05])
ax_search = plt.axes([0.15, 0.03, 0.25, 0.05])

btn_prev = Button(ax_prev, '<< Prev Page')
btn_next = Button(ax_next, 'Next Page >>')
txt_search = TextBox(ax_search, 'Search stock_id: ', initial='')

btn_prev.on_clicked(go_prev)
btn_next.on_clicked(go_next)
txt_search.on_submit(search_stock)

print("Drawing initial chart...")
draw_chart()

print("Opening interactive window...")
manager = plt.get_current_fig_manager()
try:
    manager.window.attributes('-topmost', True)
    manager.window.attributes('-topmost', False) 
except:
    pass 

plt.show()