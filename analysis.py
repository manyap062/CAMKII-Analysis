import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy import stats, signal
import seaborn as sns
import types

class CaMKIIAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("CaMKII Analysis Tool")
        self.root.geometry("1200x800")
        
        self.rhod_data = None
        self.fret_data = None
        self.time_data = None
        self.current_reading = 1

        self.rhod_normalized = {}
        self.fret_normalized = {}
        
        # data structure for automatically found peaks
        self.rhod_peaks = {}
        self.fret_peaks = {}
        self.rhod_peak_properties = {}
        self.fret_peak_properties = {}
        
        # data structure for manual peaks
        self.current_peak_index = 0
        self.manual_peak_boundaries = {}
        
        self.fret_color = 'black'
        self.rhod_color = 'red'
        self.plot_title = 'CaMKII and Calcium Analysis'
        
        # analysis parameters
        self.peak_prominence = 0.05
        self.peak_width = 5
        self.baseline_percentile = 10
        self.derivative_threshold = 0.005
        
        self.overlay_var = tk.BooleanVar(value=False)

        self.setup_ui()
    
    def setup_ui(self):
        # create main frames
        self.control_frame = ttk.Frame(self.root, padding="5")
        self.control_frame.pack(fill=tk.X)

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # add controls to load data
        tk.Button(
            self.control_frame,
            text="Load Rhod Data",
            command=self.load_rhod_data,
            height=1,
            padx=6,
            pady=2,
            bd=1,
            relief=tk.RAISED,
            highlightthickness=0
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            self.control_frame,
            text="Load FRET Data",
            command=self.load_fret_data,
            height=1,
            padx=6,
            pady=2,
            bd=1,
            relief=tk.RAISED,
            highlightthickness=0
        ).pack(side=tk.LEFT, padx=5)
        
        # trial/reading selector
        ttk.Label(self.control_frame, text="Reading:").pack(side=tk.LEFT, padx=5)
        self.reading_var = tk.StringVar(value="1")
        self.reading_selector = ttk.Spinbox(
            self.control_frame, 
            from_=1, to=20,
            textvariable=self.reading_var,
            command=self.update_plot,
            width=5
        )
        self.reading_selector.pack(side=tk.LEFT, padx=5)
        
        # save plot 
        tk.Button(
            self.control_frame,
            text="Save Plot",
            command=self.save_plot,
            height=1,
            padx=6,
            pady=2,
            bd=1,
            relief=tk.RAISED,
            highlightthickness=0
        ).pack(side=tk.LEFT, padx=5)
        
        # analysis button (amplitude, max width, rate of growth/decay, etc)
        tk.Button(
            self.control_frame,
            text="Analyze Data",
            command=self.analyze_data,
            height=1,
            padx=6,
            pady=2,
            bd=1,
            relief=tk.RAISED,
            highlightthickness=0
        ).pack(side=tk.LEFT, padx=5)

        # automatic peak detection button
        tk.Button(
            self.control_frame,
            text="Detect Peaks",
            command=self.detect_peaks,
            height=1,
            padx=6,
            pady=2,
            bd=1,
            relief=tk.RAISED,
            highlightthickness=0
        ).pack(side=tk.LEFT, padx=5)

        # reset zoom
        tk.Button(
            self.control_frame,
            text="Reset View",
            command=self.reset_view,
            height=1,
            padx=6,
            pady=2,
            bd=1,
            relief=tk.RAISED,
            highlightthickness=0
        ).pack(side=tk.LEFT, padx=5)

        # button for manual table editing view (toggles panel to right)
        self.edit_mode = False
        self.edit_button = tk.Button(
            self.control_frame,
            text="Edit Peaks",
            command=self.toggle_edit_mode,
            height=1,
            padx=6,
            pady=2,
            bd=1,
            relief=tk.RAISED,
            highlightthickness=0
        )
        self.edit_button.pack(side=tk.LEFT, padx=5)

        # clear peaks button
        tk.Button(
            self.control_frame,
            text="Clear Peaks",
            command=self.clear_peaks,
            height=1,
            padx=6,
            pady=2,
            bd=1,
            relief=tk.RAISED,
            highlightthickness=0
        ).pack(side=tk.LEFT, padx=5)

        # toggle peak boundary visibility
        self.show_boundaries_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Show Peak Boundaries", 
                      variable=self.show_boundaries_var,
                      command=self.update_plot).pack(side=tk.LEFT, padx=5)

        ttk.Checkbutton(self.control_frame, text="Merge Plots",
                      variable=self.overlay_var,
                      command=self.update_plot).pack(side=tk.LEFT, padx=5)
        
        # initialize matplotlib figure
        # embed a dedicated Figure object so tkinter does not fight pyplot state
        self.fig = Figure(figsize=(10, 8))
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        self.fig.subplots_adjust(top=0.9, hspace=0.3)

        self.ax_overlay = self.ax1.twinx()
        self.ax_overlay.set_ylabel('Rhod-2 (F/F0)')
        self.ax_overlay.set_visible(False)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # add matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self._customize_toolbar()

        # make canvas interactive for peak selection
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.root.bind('<Escape>', lambda event: self._clear_navigation_mode(update_status=True))

        # add status bar for user guidance
        self.status_bar = ttk.Label(self.root, text="Click on a plot to manually add peak markers")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self._build_edit_pane()

    def _customize_toolbar(self):
        if not hasattr(self, 'toolbar'):
            return

        original_draw = self.toolbar.draw_rubberband
        original_remove = self.toolbar.remove_rubberband

        def draw_rubberband(toolbar, event, x0, y0, x1, y1, _orig=original_draw):
            _orig(event, x0, y0, x1, y1)
            canvas = toolbar.canvas
            rect = getattr(canvas, '_rubberband_rect_black', None)
            if rect:
                canvas._tkcanvas.itemconfigure(rect, outline='red', width=2)
            white_rect = getattr(canvas, '_rubberband_rect_white', None)
            if white_rect:
                canvas._tkcanvas.delete(white_rect)
                canvas._rubberband_rect_white = None

        def remove_rubberband(toolbar, *args, _orig=original_remove):
            _orig(*args)
            canvas = toolbar.canvas
            if getattr(canvas, '_rubberband_rect_black', None):
                canvas._rubberband_rect_black = None

        self.toolbar.draw_rubberband = types.MethodType(draw_rubberband, self.toolbar)
        self.toolbar.remove_rubberband = types.MethodType(remove_rubberband, self.toolbar)

    def _build_edit_pane(self):
        self.edit_container = ttk.Frame(self.main_frame, padding="6")
        self.edit_container.pack_propagate(False)

        #styling for edit pane
        self._table_style = ttk.Style(self.root)
        self._table_style.configure('Peak.Treeview', rowheight=22, background='#ffffff', fieldbackground='#ffffff', foreground='#1f1f1f')
        self._table_style.map(
            'Peak.Treeview',
            background=[('selected', '#d9e2f3')]
        )
        self._table_style.configure('Peak.Treeview.Heading', font=('TkDefaultFont', 10, 'bold'), background='#d7dce5', foreground='#1f1f1f')
        self._table_style.map('Peak.Treeview.Heading', background=[('active', '#c8ceda')])
        self._table_style.configure('Peak.TLabelframe', background='#f2f4f8')
        #rhod and fret peak title
        self._table_style.configure('Peak.TLabelframe.Label', background='#f2f4f8', foreground='#0066cc', font=('TkDefaultFont', 10, 'bold'))

        header = ttk.Label(
            self.edit_container,
            text="Peak Editing",
            font=('TkDefaultFont', 11, 'bold')
        )
        header.pack(anchor=tk.W, pady=(0, 6))

        rhod_frame = ttk.LabelFrame(self.edit_container, text="Rhod Peaks", style='Peak.TLabelframe')
        rhod_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        fret_frame = ttk.LabelFrame(self.edit_container, text="FRET Peaks", style='Peak.TLabelframe')
        fret_frame.pack(fill=tk.BOTH, expand=True)

        columns = ('match', 'ordinal', 'time', 'action')
        headings = {
            'match': 'Match',
            'ordinal': '#',
            'time': 'Time (min)',
            'action': ''
        }

        self.rhod_tree = ttk.Treeview(
            rhod_frame,
            columns=columns,
            show='headings',
            style='Peak.Treeview',
            height=8
        )
        self.fret_tree = ttk.Treeview(
            fret_frame,
            columns=columns,
            show='headings',
            style='Peak.Treeview',
            height=8
        )

        self._table_widgets = {'Rhod': self.rhod_tree, 'FRET': self.fret_tree}

        for tree in (self.rhod_tree, self.fret_tree):
            for col in columns:
                anchor = tk.CENTER if col in ('match', 'ordinal', 'action') else tk.W
                tree.heading(col, text=headings[col])
                width = 70 if col == 'match' else 70 if col == 'time' else 60
                if col == 'action':
                    width = 50
                tree.column(col, width=width, anchor=anchor, stretch=False)
            tree.column('time', width=140, anchor=tk.CENTER)

        rhod_scroll = ttk.Scrollbar(rhod_frame, orient=tk.VERTICAL, command=self.rhod_tree.yview)
        fret_scroll = ttk.Scrollbar(fret_frame, orient=tk.VERTICAL, command=self.fret_tree.yview)
        self.rhod_tree.configure(yscroll=rhod_scroll.set)
        self.fret_tree.configure(yscroll=fret_scroll.set)

        self.rhod_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rhod_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.fret_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        fret_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        for tree, dataset in ((self.rhod_tree, 'Rhod'), (self.fret_tree, 'FRET')):
            tree.bind('<Motion>', lambda event, t=tree, d=dataset: self._on_table_motion(event, t, d))
            tree.bind('<Leave>', lambda event, t=tree, d=dataset: self._on_table_leave(t, d))
            tree.bind('<Button-1>', lambda event, t=tree, d=dataset: self._on_table_click(event, t, d))

        self._table_row_meta = {'Rhod': {}, 'FRET': {}}
        self._current_table_hover = {'Rhod': None, 'FRET': None}
        self._action_hover_row = {'Rhod': None, 'FRET': None}
        self._hover_tag = 'hover'
        self.rhod_tree.tag_configure(self._hover_tag, background='#e3e6eb', foreground='#222222')
        self.fret_tree.tag_configure(self._hover_tag, background='#e3e6eb', foreground='#222222')
        for tree in (self.rhod_tree, self.fret_tree):
            tree.tag_configure('placeholder', foreground='#888888', font=('TkDefaultFont', 9, 'italic'))

        self._action_hover_tag = 'action-hover'
        for tree in (self.rhod_tree, self.fret_tree):
            tree.tag_configure(self._action_hover_tag, foreground='#b00020')

        self._peak_highlight_artists = {'Rhod': None, 'FRET': None}

    def toggle_edit_mode(self):
        self.edit_mode = not self.edit_mode
        if self.edit_mode:
            self.edit_container.configure(width=360)
            self.edit_container.pack(side=tk.RIGHT, fill=tk.Y)
            self.edit_button.config(text="Hide Edit Pane")
            self._refresh_edit_tables()
            self.status_bar.config(text="Edit pane enabled")
        else:
            self.edit_container.pack_forget()
            self.edit_button.config(text="Edit Peaks")
            self._clear_table_highlight('Rhod')
            self._clear_table_highlight('FRET')
            self._clear_peak_highlight('Rhod')
            self._clear_peak_highlight('FRET')
            self.status_bar.config(text="Edit pane hidden")

    def _refresh_edit_tables(self):
        if not getattr(self, 'edit_mode', False):
            return

        if not hasattr(self, 'time_data') or self.time_data is None:
            self._populate_table('Rhod', [])
            self._populate_table('FRET', [])
            return

        reading = int(self.reading_var.get())
        reading_key = f'#{reading}'

        rhod_series = self.rhod_normalized.get(reading_key)
        rhod_props = self.rhod_peak_properties.get(reading_key)
        fret_series = self.fret_normalized.get(reading_key)
        fret_props = self.fret_peak_properties.get(reading_key)

        rhod_metrics = self._collect_peak_metrics(rhod_series, rhod_props or []) if rhod_series is not None else []
        fret_metrics = self._collect_peak_metrics(fret_series, fret_props or []) if fret_series is not None else []

        match_map = {'Rhod': {}, 'FRET': {}}
        matched_pairs = self._match_peak_pairs(rhod_metrics, fret_metrics)
        for idx, pair in enumerate(matched_pairs, start=1):
            match_map['Rhod'][pair['rhod']['peak_idx']] = idx
            match_map['FRET'][pair['fret']['peak_idx']] = idx

        self._populate_table('Rhod', rhod_metrics, match_map['Rhod'], reading_key)
        self._populate_table('FRET', fret_metrics, match_map['FRET'], reading_key)

    def _populate_table(self, dataset, metrics, match_map=None, reading_key=None):
        tree = self._table_widgets[dataset]
        tree.delete(*tree.get_children())
        self._table_row_meta[dataset].clear()
        self._current_table_hover[dataset] = None
        self._clear_peak_highlight(dataset, suppress_draw=True)
        self._clear_action_hover(dataset)

        if not metrics:
            tree.insert('', tk.END, values=('', '', 'No peaks detected', ''), tags=('placeholder',))
            return

        match_map = match_map or {}
        action_symbol = '✖'

        for metric in metrics:
            peak_idx = metric['peak_idx']
            match_id = match_map.get(peak_idx, '')
            time_value = float(self.time_data.iloc[peak_idx]) if self.time_data is not None else 0.0

            values = (
                str(match_id) if match_id else '',
                metric.get('ordinal', ''),
                f"{time_value:.2f}",
                action_symbol
            )
            item = tree.insert('', tk.END, values=values)
            self._table_row_meta[dataset][item] = {
                'reading_key': reading_key,
                'peak_idx': peak_idx,
                'dataset': dataset
            }

    def _on_table_motion(self, event, tree, dataset):
        row_id = tree.identify_row(event.y)
        column_id = tree.identify_column(event.x)
        columns = tree['columns']
        col_key = None
        if column_id and column_id.startswith('#'):
            index = int(column_id.replace('#', '')) - 1
            if 0 <= index < len(columns):
                col_key = columns[index]

        if not row_id or 'placeholder' in tree.item(row_id, 'tags'):
            self._clear_table_highlight(dataset)
            self._clear_action_hover(dataset)
            tree.configure(cursor='')
            return

        self._apply_table_highlight(tree, dataset, row_id)

        if col_key == 'action':
            tree.configure(cursor='hand2')
            self._apply_action_hover(tree, dataset, row_id)
        else:
            tree.configure(cursor='')
            self._clear_action_hover(dataset)

    def _on_table_leave(self, tree, dataset):
        tree.configure(cursor='')
        self._clear_table_highlight(dataset)
        self._clear_action_hover(dataset)

    def _apply_table_highlight(self, tree, dataset, item_id):
        if self._current_table_hover[dataset] == item_id:
            return

        self._clear_table_highlight(dataset, suppress_plot=True)

        existing_tags = set(tree.item(item_id, 'tags'))
        existing_tags.add(self._hover_tag)
        tree.item(item_id, tags=tuple(existing_tags))
        self._current_table_hover[dataset] = item_id

        meta = self._table_row_meta[dataset].get(item_id)
        if meta is not None:
            self._highlight_peak_on_plot(dataset, meta['reading_key'], meta['peak_idx'])

    def _clear_table_highlight(self, dataset, suppress_plot=False):
        item_id = self._current_table_hover.get(dataset)
        if item_id:
            tree = self._table_widgets[dataset]
            try:
                tags = set(tree.item(item_id, 'tags'))
                if self._hover_tag in tags:
                    tags.discard(self._hover_tag)
                    tree.item(item_id, tags=tuple(tags))
            except tk.TclError:
                pass
        self._current_table_hover[dataset] = None
        if not suppress_plot:
            self._clear_peak_highlight(dataset)

    def _apply_action_hover(self, tree, dataset, item_id):
        if self._action_hover_row[dataset] == item_id:
            return
        self._clear_action_hover(dataset)
        tags = set(tree.item(item_id, 'tags'))
        tags.add(self._action_hover_tag)
        tree.item(item_id, tags=tuple(tags))
        self._action_hover_row[dataset] = item_id

    def _clear_action_hover(self, dataset):
        current = self._action_hover_row.get(dataset)
        if current:
            tree = self._table_widgets[dataset]
            try:
                tags = set(tree.item(current, 'tags'))
                if self._action_hover_tag in tags:
                    tags.discard(self._action_hover_tag)
                    tree.item(current, tags=tuple(tags))
            except tk.TclError:
                pass
        self._action_hover_row[dataset] = None

    def _on_table_click(self, event, tree, dataset):
        row_id = tree.identify_row(event.y)
        column_id = tree.identify_column(event.x)
        if not row_id or 'placeholder' in tree.item(row_id, 'tags'):
            return

        columns = tree['columns']
        if column_id and column_id.startswith('#'):
            index = int(column_id.replace('#', '')) - 1
            if 0 <= index < len(columns) and columns[index] == 'action':
                meta = self._table_row_meta[dataset].get(row_id)
                if meta is not None:
                    self._delete_peak(dataset, meta['reading_key'], meta['peak_idx'])
                    self._clear_table_highlight(dataset)
                    return 'break'

    def _highlight_peak_on_plot(self, dataset, reading_key, peak_idx):
        if self.time_data is None:
            return

        series_dict = self.fret_normalized if dataset == 'FRET' else self.rhod_normalized
        if reading_key not in series_dict:
            return

        series = series_dict[reading_key]
        if peak_idx >= len(series):
            return

        time_value = float(self.time_data.iloc[peak_idx])
        amplitude = float(series.iloc[peak_idx])

        axis = self.ax1 if dataset == 'FRET' else (self.ax_overlay if self.overlay_var.get() else self.ax2)

        self._clear_peak_highlight(dataset, suppress_draw=True)
        highlight = axis.scatter(
            [time_value], [amplitude],
            s=100,
            facecolors='none',
            edgecolors='#d62728',
            linewidths=2,
            zorder=10
        )
        self._peak_highlight_artists[dataset] = highlight
        self.canvas.draw_idle()

    def _clear_peak_highlight(self, dataset, suppress_draw=False):
        artist = self._peak_highlight_artists.get(dataset)
        if artist is not None:
            try:
                artist.remove()
            except Exception:
                pass
            self._peak_highlight_artists[dataset] = None
            if not suppress_draw:
                self.canvas.draw_idle()

    def _delete_peak(self, dataset, reading_key, peak_idx):
        if self.time_data is None:
            return

        peaks_dict = self.rhod_peaks if dataset == 'Rhod' else self.fret_peaks
        props_dict = self.rhod_peak_properties if dataset == 'Rhod' else self.fret_peak_properties

        if reading_key not in peaks_dict:
            return

        peaks = peaks_dict[reading_key]
        indices = np.where(peaks == peak_idx)[0]
        if indices.size == 0:
            return

        idx = int(indices[0])
        peaks_dict[reading_key] = np.delete(peaks, idx)
        if reading_key in props_dict and idx < len(props_dict[reading_key]):
            del props_dict[reading_key][idx]

        if peaks_dict.get(reading_key) is not None and len(peaks_dict[reading_key]) == 0:
            peaks_dict.pop(reading_key, None)
            props_dict.pop(reading_key, None)

        self.update_plot()
        self._notify_peaks_updated()
        peak_time = float(self.time_data.iloc[peak_idx]) if peak_idx < len(self.time_data) else peak_idx
        self.status_bar.config(text=f"Removed {dataset} peak at {peak_time:.2f} min")

    def _notify_peaks_updated(self):
        if getattr(self, 'edit_mode', False):
            self._refresh_edit_tables()

    def _clear_navigation_mode(self, update_status=False):
        if not hasattr(self, 'toolbar'):
            return

        try:
            if getattr(self.canvas, 'widgetlock', None) is not None:
                try:
                    self.canvas.widgetlock.release(self.toolbar)
                except Exception:
                    pass

            if getattr(self.toolbar, '_active', None) is not None:
                self.toolbar._active = None

            if getattr(self.toolbar, 'mode', ''):
                self.toolbar.mode = ''

            if hasattr(self.toolbar, '_update_buttons_checked'):
                self.toolbar._update_buttons_checked()

            if update_status:
                self.status_bar.config(text="Pan/Zoom tools disabled")
        except Exception as exc:
            print(f"Warning clearing navigation mode: {exc}")

    def reset_view(self):
        self._clear_navigation_mode()

        try:
            if hasattr(self, 'toolbar') and self.toolbar is not None:
                self.toolbar.home()
                self.status_bar.config(text="View reset to default")
                return
        except Exception as exc:
            print(f"Warning resetting view via toolbar: {exc}")

        if self.time_data is None:
            return

        x_min = float(self.time_data.min())
        x_max = float(self.time_data.max())
        self.ax1.set_xlim(x_min, x_max)
        if self.overlay_var.get():
            self.ax_overlay.set_xlim(x_min, x_max)
        else:
            self.ax2.set_xlim(x_min, x_max)

        for axis in (self.ax1, self.ax2, self.ax_overlay):
            axis.relim()
            axis.autoscale_view()

        self.canvas.draw_idle()
        self.status_bar.config(text="View reset to default")
    
    def load_rhod_data(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Excel files", "*.xlsx *.xls")]
            )
            if file_path:
                self.rhod_data = pd.read_excel(file_path, sheet_name = "Raw Data")
                self._update_time_axis(self.rhod_data, source_label="Rhod")
                self.normalize_rhod_data()
                self.update_plot()
                messagebox.showinfo("Success", "Rhod data loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Rhod data: {str(e)}")

    def load_fret_data(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Excel files", "*.xlsx *.xls")]
            )
            if file_path:
                self.fret_data = pd.read_excel(file_path, sheet_name="Raw Data")
                self._update_time_axis(self.fret_data, source_label="FRET")
                self.normalize_fret_data()
                self.update_plot()
                messagebox.showinfo("Success", "FRET data loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load FRET data: {str(e)}")

    def _update_time_axis(self, dataframe, source_label=""):
        """grab the shared time column and sanity check alignment"""
        if dataframe is None:
            return

        time_column = None
        for candidate in ["Time [ms]", "Time (ms)", "Time", "Minutes"]:
            if candidate in dataframe.columns:
                time_column = dataframe[candidate].astype(float)
                if "ms" in candidate.lower():
                    time_column = time_column / 60000.0
                break

        if time_column is None:
            raise ValueError(f"Time column missing in {source_label} file")

        if self.time_data is None:
            self.time_data = time_column.reset_index(drop=True)
            return

        if len(time_column) != len(self.time_data):
            messagebox.showwarning(
                "Time Axis Mismatch",
                f"{source_label} time axis length differs from the current data. Using the original timeline."
            )
            return

        if not np.allclose(time_column.values, self.time_data.values, atol=1e-6):
            messagebox.showwarning(
                "Time Axis Mismatch",
                f"{source_label} time axis values differ from the current data. Using the original timeline."
            )

    def _estimate_points_per_minute(self):
        if self.time_data is None or len(self.time_data) < 2:
            return 1.0

        diffs = np.diff(self.time_data.values)
        positive_diffs = diffs[diffs > 0]
        if len(positive_diffs) == 0:
            return 1.0

        median_step = np.median(positive_diffs)
        if median_step <= 0:
            return 1.0

        return 1.0 / median_step

    def _rise_decay_times(self, data_series, peak_idx, left_idx, right_idx, baseline):
        time_series = self.time_data
        peak_value = float(data_series.iloc[peak_idx])
        amplitude = peak_value - baseline
        if amplitude <= 0:
            return 0.0, 0.0

        threshold = baseline + 0.1 * amplitude

        rise_idx = left_idx
        for idx in range(left_idx, peak_idx + 1):
            if data_series.iloc[idx] >= threshold:
                rise_idx = idx
                break

        decay_idx = right_idx
        for idx in range(peak_idx, right_idx + 1):
            if data_series.iloc[idx] <= threshold:
                decay_idx = idx
                break

        rise_time = float(time_series.iloc[peak_idx] - time_series.iloc[rise_idx])
        decay_time = float(time_series.iloc[decay_idx] - time_series.iloc[peak_idx])

        return max(rise_time, 0.0), max(decay_time, 0.0)

    def _collect_peak_metrics(self, data_series, peak_properties):
        metrics = []
        if self.time_data is None:
            return metrics

        for idx, props in enumerate(peak_properties):
            peak_idx = props['peak_idx']
            left_idx = max(0, props['left_base'])
            right_idx = min(len(data_series) - 1, props['right_base'])

            baseline_candidates = [
                props.get('baseline'),
                float(data_series.iloc[left_idx]),
                float(data_series.iloc[right_idx])
            ]
            baseline_values = [val for val in baseline_candidates if val is not None and not np.isnan(val)]
            baseline = float(min(baseline_values)) if baseline_values else float(data_series.min())

            peak_value = float(data_series.iloc[peak_idx])
            amplitude = max(0.0, peak_value - baseline)

            time_segment = self.time_data.iloc[left_idx:right_idx+1].values
            signal_segment = data_series.iloc[left_idx:right_idx+1].values
            auc = np.trapz(signal_segment - baseline, time_segment)
            auc = max(0.0, float(auc))

            rise_time, decay_time = self._rise_decay_times(data_series, peak_idx, left_idx, right_idx, baseline)
            width = float(self.time_data.iloc[right_idx] - self.time_data.iloc[left_idx])

            metrics.append({
                'ordinal': idx + 1,
                'peak_idx': peak_idx,
                'left_idx': left_idx,
                'right_idx': right_idx,
                'baseline': baseline,
                'peak_value': peak_value,
                'amplitude': amplitude,
                'auc': auc,
                'rise_time': rise_time,
                'decay_time': decay_time,
                'width': width
            })

        return metrics

    def _match_peak_pairs(self, rhod_metrics, fret_metrics):
        if not rhod_metrics or not fret_metrics:
            return []

        rhod_sorted = sorted(rhod_metrics, key=lambda m: self.time_data.iloc[m['peak_idx']])
        fret_sorted = sorted(fret_metrics, key=lambda m: self.time_data.iloc[m['peak_idx']])
        used_fret = set()
        pairs = []

        for rhod_metric in rhod_sorted:
            rhod_time = float(self.time_data.iloc[rhod_metric['peak_idx']])
            best_candidate = None
            best_delay = None

            for fret_metric in fret_sorted:
                if fret_metric['peak_idx'] in used_fret:
                    continue

                fret_time = float(self.time_data.iloc[fret_metric['peak_idx']])
                if fret_time <= rhod_time:
                    continue

                delay = fret_time - rhod_time
                if best_delay is None or delay < best_delay:
                    best_delay = delay
                    best_candidate = fret_metric

            if best_candidate is not None:
                used_fret.add(best_candidate['peak_idx'])
                pairs.append({
                    'rhod': rhod_metric,
                    'fret': best_candidate,
                    'delay': best_delay
                })

        return pairs
    
    def normalize_rhod_data(self):
        if self.rhod_data is not None:
            self.rhod_normalized.clear()

            for col in self.rhod_data.columns:
                if col.startswith('#'):
                    try:
                        reading_num = col.split()[0].replace('#', '')
                        average = self.rhod_data[col].iloc[:6].mean()
                        normalized = self.rhod_data[col] / average
                        self.rhod_normalized[f'#{reading_num}'] = normalized
                    except Exception as e:
                        print(f"Error normalizing column {col}: {str(e)}")

    
    def normalize_fret_data(self):
        if self.fret_data is not None:
            self.fret_normalized.clear()

            for col in self.fret_data.columns:
                if col.startswith('#') and not col.endswith('_norm'):
                    try:
                        reading_num = col.split()[0].replace('#', '')
                        inverted = 1 / self.fret_data[col]
                        # Normalize by first 6 readings
                        average = inverted.iloc[:6].mean()
                        normalized = inverted/average
                        self.fret_normalized[f'#{reading_num}'] = normalized

                    except Exception as e:
                        print(f"Error normalizing column {col}: {str(e)}")
    
    def update_plot(self):
        reading = int(self.reading_var.get())
        reading_key = f'#{reading}'
        overlay_mode = self.overlay_var.get()

        self._clear_peak_highlight('Rhod', suppress_draw=True)
        self._clear_peak_highlight('FRET', suppress_draw=True)

        self.ax1.clear()
        self.ax1.set_ylabel('FRET change')

        if overlay_mode:
            self.ax_overlay.set_visible(True)
            self.ax_overlay.clear()
            self.ax_overlay.set_ylabel('Rhod-2 (F/F0)')
            self.ax2.clear()
            self.ax2.set_visible(False)
        else:
            self.ax_overlay.clear()
            self.ax_overlay.set_visible(False)
            self.ax2.set_visible(True)
            self.ax2.clear()
            self.ax2.set_ylabel('Rhod-2 (F/F0)')
            self.ax2.set_xlabel('Time (min)')

        if self.time_data is None:
            self.ax1.text(0.5, 0.5, 'Load data to start', transform=self.ax1.transAxes,
                          ha='center', va='center', fontsize=12, color='gray')
            if overlay_mode:
                self.ax1.set_xlabel('Time (min)')
            else:
                self.ax2.text(0.5, 0.5, 'Waiting for data', transform=self.ax2.transAxes,
                              ha='center', va='center', fontsize=12, color='gray')
            self.status_bar.config(text="Load a dataset to start plotting")
            self.canvas.draw_idle()
            return

        has_fret = reading_key in self.fret_normalized
        has_rhod = reading_key in self.rhod_normalized
        rhod_axis = self.ax_overlay if overlay_mode else self.ax2
        plotted_any = False

        if has_fret:
            fret_series = self.fret_normalized[reading_key]
            self.ax1.plot(self.time_data, fret_series, color=self.fret_color, label='FRET')
            plotted_any = True
        else:
            self.ax1.text(0.5, 0.5, 'Reading missing in FRET file', transform=self.ax1.transAxes,
                          ha='center', va='center', fontsize=12, color='gray')

        if has_rhod:
            rhod_series = self.rhod_normalized[reading_key]
            rhod_axis.plot(self.time_data, rhod_series, color=self.rhod_color, label='Rhod-2')
            if not overlay_mode:
                rhod_axis.set_xlabel('Time (min)')
            plotted_any = True
        else:
            rhod_axis.text(0.5, 0.5, 'Reading missing in Rhod file', transform=rhod_axis.transAxes,
                           ha='center', va='center', fontsize=12, color='gray')

        boundary_color = '#4c72b0'

        if has_rhod and reading_key in self.rhod_peaks:
            rhod_peaks = self.rhod_peaks[reading_key]
            rhod_axis.plot(self.time_data[rhod_peaks],
                           rhod_series.iloc[rhod_peaks],
                           'x', color=boundary_color, markersize=7,
                           label=f'Rhod peaks ({len(rhod_peaks)})')

            if self.show_boundaries_var.get() and reading_key in self.rhod_peak_properties:
                for props in self.rhod_peak_properties[reading_key]:
                    left_time = self.time_data[props['left_base']]
                    right_time = self.time_data[props['right_base']]
                    rhod_axis.axvline(x=left_time, color=boundary_color, linestyle='--', alpha=0.35)
                    rhod_axis.axvline(x=right_time, color=boundary_color, linestyle='--', alpha=0.35)

        if has_fret and reading_key in self.fret_peaks:
            fret_peaks = self.fret_peaks[reading_key]
            self.ax1.plot(self.time_data[fret_peaks],
                          fret_series.iloc[fret_peaks],
                          'x', color=boundary_color, markersize=7,
                          label=f'FRET peaks ({len(fret_peaks)})')

            if self.show_boundaries_var.get() and reading_key in self.fret_peak_properties:
                for props in self.fret_peak_properties[reading_key]:
                    left_time = self.time_data[props['left_base']]
                    right_time = self.time_data[props['right_base']]
                    self.ax1.axvline(x=left_time, color=boundary_color, linestyle='--', alpha=0.35)
                    self.ax1.axvline(x=right_time, color=boundary_color, linestyle='--', alpha=0.35)

        if overlay_mode:
            self.ax1.set_xlabel('Time (min)')
            handles1, labels1 = self.ax1.get_legend_handles_labels()
            handles2, labels2 = rhod_axis.get_legend_handles_labels()
            if handles1 or handles2:
                legend_map = {label: handle for handle, label in zip(handles1 + handles2, labels1 + labels2)}
                legend_handles = list(legend_map.values())
                legend_labels = list(legend_map.keys())
                self.ax1.legend(legend_handles, legend_labels, loc='upper right')
        else:
            self.ax1.set_xlabel('')
            handles1, labels1 = self.ax1.get_legend_handles_labels()
            if handles1:
                legend_map = {label: handle for handle, label in zip(handles1, labels1)}
                legend_handles = list(legend_map.values())
                legend_labels = list(legend_map.keys())
                self.ax1.legend(legend_handles, legend_labels)

            handles2, labels2 = rhod_axis.get_legend_handles_labels()
            if handles2:
                legend_map = {label: handle for handle, label in zip(handles2, labels2)}
                legend_handles = list(legend_map.values())
                legend_labels = list(legend_map.keys())
                rhod_axis.legend(legend_handles, legend_labels)

        self.fig.suptitle(self.plot_title)
        self.canvas.draw_idle()

        if plotted_any:
            self.status_bar.config(text=f"Showing reading #{reading}")
        else:
            self.status_bar.config(text="Reading not available in loaded files")

        if getattr(self, 'edit_mode', False):
            self._refresh_edit_tables()

    def save_plot(self):
        if self.time_data is None:
            messagebox.showwarning("Warning", "Nothing to save yet. Load data first.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Success", f"Plot saved to {file_path}")
    
    def detect_peaks(self):
        if self.rhod_data is None and self.fret_data is None:
            messagebox.showwarning("Warning", "Please load both Rhod and FRET data first.")
            return

        if self.rhod_data is None:
            messagebox.showwarning("Warning", "Please load Rhod data first.")
            return

        if self.fret_data is None:
            messagebox.showwarning("Warning", "Please load FRET data first.")
            return

        reading = int(self.reading_var.get())
        reading_key = f'#{reading}'

        missing_sources = []
        if reading_key not in self.rhod_normalized:
            missing_sources.append("Rhod")
        if reading_key not in self.fret_normalized:
            missing_sources.append("FRET")

        if missing_sources:
            label = " and ".join(missing_sources)
            messagebox.showwarning("Warning", f"Reading {reading} not found in {label} data.")
            return

        rhod_data = self.rhod_normalized[reading_key]
        fret_data = self.fret_normalized[reading_key]

        rhod_peaks, rhod_props = signal.find_peaks(
            rhod_data,
            height=1.05,
            distance=10,
            prominence=0.05,
            width=3
        )

        fret_peaks, fret_props = signal.find_peaks(
            fret_data,
            height=1.005,
            distance=5,
            prominence=0.003,
            width=2
        )

        rhod_peak_properties = []
        for i in range(len(rhod_peaks)):
            properties = {
                'peak_idx': int(rhod_peaks[i]),
                'peak_height': float(rhod_props['peak_heights'][i]),
                'left_base': int(rhod_props['left_bases'][i]),
                'right_base': int(rhod_props['right_bases'][i]),
                'prominence': float(rhod_props['prominences'][i]),
                'width': float(rhod_props['widths'][i])
            }
            rhod_peak_properties.append(properties)

        fret_peak_properties = []
        for i in range(len(fret_peaks)):
            properties = {
                'peak_idx': int(fret_peaks[i]),
                'peak_height': float(fret_props['peak_heights'][i]),
                'left_base': int(fret_props['left_bases'][i]),
                'right_base': int(fret_props['right_bases'][i]),
                'prominence': float(fret_props['prominences'][i]),
                'width': float(fret_props['widths'][i])
            }
            fret_peak_properties.append(properties)

        self.rhod_peaks[reading_key] = np.array(rhod_peaks, dtype=int)
        self.fret_peaks[reading_key] = np.array(fret_peaks, dtype=int)
        self.rhod_peak_properties[reading_key] = rhod_peak_properties
        self.fret_peak_properties[reading_key] = fret_peak_properties

        self.update_plot()

        summary = f"Detected {len(rhod_peaks)} Rhod peaks and {len(fret_peaks)} FRET peaks."
        self.status_bar.config(text=summary)
        messagebox.showinfo("Peak Detection", summary)
        self._notify_peaks_updated()

    def clear_peaks(self):
        """Clear all detected peaks for the current reading"""
        reading = int(self.reading_var.get())
        reading_key = f'#{reading}'
        
        if reading_key in self.rhod_peaks:
            del self.rhod_peaks[reading_key]
        if reading_key in self.fret_peaks:
            del self.fret_peaks[reading_key]
        if reading_key in self.rhod_peak_properties:
            del self.rhod_peak_properties[reading_key]
        if reading_key in self.fret_peak_properties:
            del self.fret_peak_properties[reading_key]
            
        self.update_plot()
        self.status_bar.config(text=f"All peaks cleared for Reading {reading}")
        self._notify_peaks_updated()
    
    def on_click(self, event):
        """Handle plot clicks for manual peak editing."""
        if event.inaxes is None or event.xdata is None:
            return

        reading = int(self.reading_var.get())
        reading_key = f'#{reading}'

        if event.button == 3:
            self._remove_peak_via_click(event, reading_key)
            return

        key = (event.key or '').lower()
        if event.button == 1 and 'shift' in key:
            self._add_peak_via_click(event, reading_key)

    def _remove_peak_via_click(self, event, reading_key):
        self._clear_navigation_mode()
        target_dicts = None
        label = ""

        if event.inaxes == self.ax1:
            target_dicts = (self.fret_peaks, self.fret_peak_properties, self.fret_normalized, 'FRET')
        elif event.inaxes in (self.ax2, self.ax_overlay):
            target_dicts = (self.rhod_peaks, self.rhod_peak_properties, self.rhod_normalized, 'Rhod')

        if target_dicts is None:
            return

        peaks_dict, props_dict, data_dict, label = target_dicts
        if reading_key not in peaks_dict or reading_key not in data_dict:
            return

        peaks = peaks_dict[reading_key]
        props = props_dict[reading_key]
        if len(peaks) == 0:
            return

        clicked_idx = int(np.abs(self.time_data - event.xdata).argmin())
        tolerance = max(1, int(self._estimate_points_per_minute() * 0.7))
        distances = np.abs(peaks - clicked_idx)
        nearest_idx = int(np.argmin(distances))

        if distances[nearest_idx] <= tolerance:
            peaks_dict[reading_key] = np.delete(peaks, nearest_idx)
            del props[nearest_idx]
            if len(peaks_dict[reading_key]) == 0:
                peaks_dict.pop(reading_key, None)
                props_dict.pop(reading_key, None)
            self.status_bar.config(
                text=f"Removed {label} peak at {self.time_data[clicked_idx]:.2f} min"
            )
            self.update_plot()
            self._notify_peaks_updated()

    def _add_peak_via_click(self, event, reading_key):
        self._clear_navigation_mode()
        target_dicts = None

        if event.inaxes == self.ax1:
            target_dicts = (self.fret_peaks, self.fret_peak_properties, self.fret_normalized, 'FRET')
        elif event.inaxes in (self.ax2, self.ax_overlay):
            target_dicts = (self.rhod_peaks, self.rhod_peak_properties, self.rhod_normalized, 'Rhod')

        if target_dicts is None:
            return

        peaks_dict, props_dict, data_dict, label = target_dicts
        if reading_key not in data_dict:
            messagebox.showwarning("Manual Peak", f"{label} data not loaded for this reading.")
            return

        series = data_dict[reading_key]
        clicked_idx = int(np.abs(self.time_data - event.xdata).argmin())

        if reading_key not in peaks_dict:
            peaks_dict[reading_key] = np.array([], dtype=int)
            props_dict[reading_key] = []

        peaks = peaks_dict[reading_key]
        props = props_dict[reading_key]

        if clicked_idx in peaks:
            self.status_bar.config(
                text=f"{label} peak already exists at {self.time_data[clicked_idx]:.2f} min"
            )
            return

        peak_value = float(series.iloc[clicked_idx])
        left_base = max(0, clicked_idx - 20)
        right_base = min(len(series) - 1, clicked_idx + 20)

        new_peaks = np.append(peaks, clicked_idx)
        order = np.argsort(new_peaks)
        peaks_dict[reading_key] = new_peaks[order].astype(int)

        new_props = props + [{
            'peak_idx': clicked_idx,
            'peak_height': peak_value,
            'left_base': left_base,
            'right_base': right_base,
            'prominence': peak_value - 1.0,
            'width': 10.0
        }]
        props_dict[reading_key] = [new_props[idx] for idx in order]

        self.status_bar.config(text=f"Added {label} peak at {self.time_data[clicked_idx]:.2f} min")
        self.update_plot()
        self._notify_peaks_updated()
    
    def analyze_data(self):
        # Get the current reading from the UI
        reading = int(self.reading_var.get())
        reading_key = f'#{reading}'

        rhod_available = reading_key in self.rhod_normalized
        fret_available = reading_key in self.fret_normalized

        if not rhod_available and not fret_available:
            messagebox.showwarning("Warning", f"No data loaded for reading {reading}.")
            return
            
        peaks_present = False
        if rhod_available:
            peaks_present = peaks_present or (reading_key in self.rhod_peaks and len(self.rhod_peaks[reading_key]) > 0)
        if fret_available:
            peaks_present = peaks_present or (reading_key in self.fret_peaks and len(self.fret_peaks[reading_key]) > 0)

        if not peaks_present:
            result = messagebox.askyesno("No Peaks Detected", 
                                        f"No peaks have been detected for Reading #{reading}. Would you like to detect peaks now?")
            if result:
                self.detect_peaks()
            else:
                return

        # Create analysis window
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title(f"Analysis Results - Reading #{reading}")
        analysis_window.geometry("800x600")
        
        # Create notebook with tabs
        notebook = ttk.Notebook(analysis_window)
        
        # Summary tab
        summary_tab = ttk.Frame(notebook)
        notebook.add(summary_tab, text="Summary")
        
        # Peaks tab
        peaks_tab = ttk.Frame(notebook)
        notebook.add(peaks_tab, text="Peak Details")
        
        # Correlation tab for Ca-CaMKII relationship
        correlation_tab = ttk.Frame(notebook)
        notebook.add(correlation_tab, text="Ca-CaMKII Correlation")
        
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Summary results
        summary_text = tk.Text(summary_tab, wrap=tk.WORD, width=80, height=30)
        summary_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        rhod_data = self.rhod_normalized[reading_key] if rhod_available else None
        fret_data = self.fret_normalized[reading_key] if fret_available else None

        rhod_metrics = self._collect_peak_metrics(
            rhod_data, self.rhod_peak_properties.get(reading_key, [])
        ) if rhod_available else []

        fret_metrics = self._collect_peak_metrics(
            fret_data, self.fret_peak_properties.get(reading_key, [])
        ) if fret_available else []

        # Calculate overall statistics
        rhod_baseline = np.percentile(rhod_data, self.baseline_percentile) if rhod_available else None
        fret_baseline = np.percentile(fret_data, self.baseline_percentile) if fret_available else None

        # Overall summary
        summary_text.insert(tk.END, f"Analysis Results for Reading #{reading}\n\n")
        summary_text.insert(tk.END, f"Rhod-2 (Calcium) Statistics:\n")
        if rhod_available:
            summary_text.insert(tk.END, f"- Number of peaks: {len(rhod_metrics)}\n")
            summary_text.insert(tk.END, f"- Overall baseline: {rhod_baseline:.3f}\n")
            summary_text.insert(tk.END, f"- Max value: {rhod_data.max():.3f}\n")
            if rhod_metrics:
                avg_rise = np.mean([m['rise_time'] for m in rhod_metrics])
                avg_decay = np.mean([m['decay_time'] for m in rhod_metrics])
                avg_auc = np.mean([m['auc'] for m in rhod_metrics])
                summary_text.insert(tk.END, f"- Average rise time: {avg_rise:.2f} min\n")
                summary_text.insert(tk.END, f"- Average decay time: {avg_decay:.2f} min\n")
                summary_text.insert(tk.END, f"- Mean AUC (baseline-subtracted): {avg_auc:.3f}\n")
        else:
            summary_text.insert(tk.END, "- Rhod-2 data not loaded\n")

        summary_text.insert(tk.END, f"\nFRET (CaMKII) Statistics:\n")
        if fret_available:
            summary_text.insert(tk.END, f"- Number of peaks: {len(fret_metrics)}\n")
            summary_text.insert(tk.END, f"- Overall baseline: {fret_baseline:.3f}\n")
            summary_text.insert(tk.END, f"- Max value: {fret_data.max():.3f}\n")
            if fret_metrics:
                avg_rise = np.mean([m['rise_time'] for m in fret_metrics])
                avg_decay = np.mean([m['decay_time'] for m in fret_metrics])
                avg_auc = np.mean([m['auc'] for m in fret_metrics])
                summary_text.insert(tk.END, f"- Average rise time: {avg_rise:.2f} min\n")
                summary_text.insert(tk.END, f"- Average decay time: {avg_decay:.2f} min\n")
                summary_text.insert(tk.END, f"- Mean AUC (baseline-subtracted): {avg_auc:.3f}\n")
        else:
            summary_text.insert(tk.END, "- FRET data not loaded\n")

        matched_pairs = self._match_peak_pairs(rhod_metrics, fret_metrics)
        if matched_pairs:
            delays = [pair['delay'] for pair in matched_pairs]
            summary_text.insert(tk.END, f"\nComparative Analysis:\n")
            summary_text.insert(tk.END, f"- Matches: {len(matched_pairs)}\n")
            summary_text.insert(tk.END, f"- Average Ca ➜ CaMKII delay: {np.mean(delays):.2f} min\n")
            summary_text.insert(tk.END, f"- Min delay: {np.min(delays):.2f} min\n")
            summary_text.insert(tk.END, f"- Max delay: {np.max(delays):.2f} min\n")

        # peak details tab
        peak_details = ttk.Frame(peaks_tab)
        peak_details.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # create tables for peak details
        self.create_peak_details_table(peak_details, reading_key, rhod_metrics, fret_metrics)

        # correlation tab
        if matched_pairs:
            self.create_correlation_analysis(correlation_tab, matched_pairs)
        else:
            ttk.Label(correlation_tab, text="Insufficient peaks for correlation analysis").pack(pady=20)
        
        # add save table button
        ttk.Button(analysis_window, text="Save Analysis", 
                  command=lambda: self.save_analysis(summary_text.get("1.0", tk.END))).pack(pady=10)
    
    def create_peak_details_table(self, parent, reading_key, rhod_metrics, fret_metrics):
        # create separate frames for Rhod and FRET
        rhod_frame = ttk.LabelFrame(parent, text="Rhod-2 (Calcium) Peaks")
        rhod_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        fret_frame = ttk.LabelFrame(parent, text="FRET (CaMKII) Peaks")
        fret_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # rhod peaks table
        if rhod_metrics:
            columns = ("Peak #", "Position", "Amplitude", "Width", "AUC", "Rise Time", "Decay Time")
            rhod_tree = ttk.Treeview(rhod_frame, columns=columns, show="headings")
            
            for col in columns:
                rhod_tree.heading(col, text=col)
                rhod_tree.column(col, width=100, anchor=tk.CENTER)

            for metric in rhod_metrics:
                peak_time = float(self.time_data.iloc[metric['peak_idx']])
                rhod_tree.insert("", tk.END, values=(
                    metric['ordinal'],
                    f"{peak_time:.2f}",
                    f"{metric['amplitude']:.3f}",
                    f"{metric['width']:.2f}",
                    f"{metric['auc']:.3f}",
                    f"{metric['rise_time']:.2f}",
                    f"{metric['decay_time']:.2f}"
                ))
            
            scrollbar = ttk.Scrollbar(rhod_frame, orient=tk.VERTICAL, command=rhod_tree.yview)
            rhod_tree.configure(yscroll=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            rhod_tree.pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(rhod_frame, text="No Rhod peaks detected").pack(pady=20)
        
        # fret peaks table
        if fret_metrics:
            columns = ("Peak #", "Position", "Amplitude", "Width", "AUC", "Rise Time", "Decay Time")
            fret_tree = ttk.Treeview(fret_frame, columns=columns, show="headings")
            
            for col in columns:
                fret_tree.heading(col, text=col)
                fret_tree.column(col, width=100, anchor=tk.CENTER)

            for metric in fret_metrics:
                peak_time = float(self.time_data.iloc[metric['peak_idx']])
                fret_tree.insert("", tk.END, values=(
                    metric['ordinal'],
                    f"{peak_time:.2f}",
                    f"{metric['amplitude']:.3f}",
                    f"{metric['width']:.2f}",
                    f"{metric['auc']:.3f}",
                    f"{metric['rise_time']:.2f}",
                    f"{metric['decay_time']:.2f}"
                ))
            
            scrollbar = ttk.Scrollbar(fret_frame, orient=tk.VERTICAL, command=fret_tree.yview)
            fret_tree.configure(yscroll=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            fret_tree.pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(fret_frame, text="No FRET peaks detected").pack(pady=20)
    
    #TODO: decide if we even need this - flawed assumption
    def create_correlation_analysis(self, parent, matched_pairs):
        # create a matplotlib figure for correlation plotting
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        if matched_pairs:
            df = pd.DataFrame({
                'Ca_amplitude': [pair['rhod']['amplitude'] for pair in matched_pairs],
                'CaMKII_amplitude': [pair['fret']['amplitude'] for pair in matched_pairs],
                'Ca_AUC': [pair['rhod']['auc'] for pair in matched_pairs],
                'CaMKII_AUC': [pair['fret']['auc'] for pair in matched_pairs],
                'Delay': [pair['delay'] for pair in matched_pairs]
            })

            ax.scatter(df['Ca_amplitude'], df['CaMKII_amplitude'],
                      c=df['Delay'], cmap='viridis', s=100, alpha=0.7)
            ax.set_xlabel('Ca²⁺ Peak Amplitude')
            ax.set_ylabel('CaMKII Peak Amplitude')
            ax.set_title('Ca²⁺ vs CaMKII Peak Amplitude Correlation')
            
            # Add best fit line
            if len(df) > 1:  # Need at least 2 points for regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df['Ca_amplitude'], df['CaMKII_amplitude'])
                x = np.array([df['Ca_amplitude'].min(), df['Ca_amplitude'].max()])
                y = slope * x + intercept
                ax.plot(x, y, 'r--', 
                       label=f'y = {slope:.2f}x + {intercept:.2f}\nr² = {r_value**2:.2f}')
                ax.legend()
            
            # Add colorbar for delay
            cbar = fig.colorbar(ax.collections[0])
            cbar.set_label('Delay (min)')
            
            # Add the plot to the tab
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Add correlation statistics
            stats_frame = ttk.LabelFrame(parent, text="Correlation Statistics")
            stats_frame.pack(fill=tk.X, padx=10, pady=5)
            
            stats_text = tk.Text(stats_frame, height=8, width=80)
            stats_text.pack(padx=5, pady=5, fill=tk.BOTH)
            
            # Calculate correlations
            amp_corr = df['Ca_amplitude'].corr(df['CaMKII_amplitude'])
            auc_corr = df['Ca_AUC'].corr(df['CaMKII_AUC'])
            
            stats_text.insert(tk.END, f"Number of matched peaks: {len(df)}\n")
            stats_text.insert(tk.END, f"Amplitude correlation (r): {amp_corr:.3f}\n")
            stats_text.insert(tk.END, f"AUC correlation (r): {auc_corr:.3f}\n")
            stats_text.insert(tk.END, f"Average delay: {df['Delay'].mean():.2f} min\n")
            stats_text.insert(tk.END, f"Std. deviation of delay: {df['Delay'].std():.2f} min\n")
            
            if len(df) > 1 and not np.isnan(amp_corr):
                # Linear regression stats for amplitude correlation
                stats_text.insert(tk.END, f"\nLinear regression:\n")
                stats_text.insert(tk.END, f"Slope: {slope:.3f}\n")
                stats_text.insert(tk.END, f"Intercept: {intercept:.3f}\n")
                stats_text.insert(tk.END, f"R-squared: {r_value**2:.3f}\n")
                stats_text.insert(tk.END, f"p-value: {p_value:.4f}\n")
        else:
            ttk.Label(parent, text="Could not match Ca and CaMKII peaks for correlation analysis").pack(pady=20)
    
    def save_analysis(self, analysis_text):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w') as f:
                f.write(analysis_text)
            messagebox.showinfo("Success", f"Analysis saved to {file_path}")
    
    #TODO: make this whole thing work to adjust peak boundaries manually
    def adjust_peak_boundaries(self):
        reading = int(self.reading_var.get())
        reading_key = f'#{reading}'
        
        # Check if we have peaks detected
        if (reading_key not in self.rhod_peaks or len(self.rhod_peaks[reading_key]) == 0) and \
           (reading_key not in self.fret_peaks or len(self.fret_peaks[reading_key]) == 0):
            messagebox.showwarning("No Peaks", "Please detect peaks first")
            return
        
        # Create a new window for peak boundary adjustment
        adjustment_window = tk.Toplevel(self.root)
        adjustment_window.title(f"Adjust Peak Boundaries - Reading #{reading}")
        adjustment_window.geometry("600x400")
        
        # Create notebook with tabs for FRET and Rhod
        notebook = ttk.Notebook(adjustment_window)
        fret_tab = ttk.Frame(notebook)
        rhod_tab = ttk.Frame(notebook)
        notebook.add(fret_tab, text="FRET Peaks")
        notebook.add(rhod_tab, text="Rhod Peaks")
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # FRET peaks adjustment controls
        if reading_key in self.fret_peaks and len(self.fret_peaks[reading_key]) > 0:
            # Peak selector
            ttk.Label(fret_tab, text="Select Peak:").grid(row=0, column=0, padx=5, pady=5)
            fret_peak_var = tk.IntVar(value=1)
            fret_peak_combo = ttk.Combobox(
                fret_tab,
                textvariable=fret_peak_var,
                values=[str(i+1) for i in range(len(self.fret_peaks[reading_key]))],
                state="readonly",
                width=5
            )
            fret_peak_combo.grid(row=0, column=1, padx=5, pady=5)
            
            # Boundary adjustment sliders
            ttk.Label(fret_tab, text="Left Boundary:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            fret_left_var = tk.IntVar(value=0)
            fret_left_slider = ttk.Scale(
                fret_tab,
                from_=0,
                to=len(self.time_data)-1,
                orient=tk.HORIZONTAL,
                length=200,
                variable=fret_left_var
            )
            fret_left_slider.grid(row=1, column=1, padx=5, pady=5)
            
            ttk.Label(fret_tab, text="Right Boundary:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
            fret_right_var = tk.IntVar(value=100)
            fret_right_slider = ttk.Scale(
                fret_tab,
                from_=0,
                to=len(self.time_data)-1,
                orient=tk.HORIZONTAL,
                length=200,
                variable=fret_right_var
            )
            fret_right_slider.grid(row=2, column=1, padx=5, pady=5)
            
            # Update button
            ttk.Button(
                fret_tab,
                text="Update Boundaries",
                command=lambda: self.update_boundaries("fret", int(fret_peak_var.get())-1, fret_left_var.get(), fret_right_var.get())
            ).grid(row=3, column=0, columnspan=2, pady=10)
            
            # Function to update sliders when peak selection changes
            def update_fret_sliders(*args):
                peak_idx = int(fret_peak_var.get()) - 1
                if 0 <= peak_idx < len(self.fret_peak_properties[reading_key]):
                    props = self.fret_peak_properties[reading_key][peak_idx]
                    fret_left_var.set(props['left_base'])
                    fret_right_var.set(props['right_base'])
            
            fret_peak_combo.bind("<<ComboboxSelected>>", update_fret_sliders)
            update_fret_sliders()  # Initialize with first peak
        else:
            ttk.Label(fret_tab, text="No FRET peaks detected").pack(pady=20)
        
        # Rhod peaks adjustment controls
        if reading_key in self.rhod_peaks and len(self.rhod_peaks[reading_key]) > 0:
            # Peak selector
            ttk.Label(rhod_tab, text="Select Peak:").grid(row=0, column=0, padx=5, pady=5)
            rhod_peak_var = tk.IntVar(value=1)
            rhod_peak_combo = ttk.Combobox(
                rhod_tab,
                textvariable=rhod_peak_var,
                values=[str(i+1) for i in range(len(self.rhod_peaks[reading_key]))],
                state="readonly",
                width=5
            )
            rhod_peak_combo.grid(row=0, column=1, padx=5, pady=5)
            
            # Boundary adjustment sliders
            ttk.Label(rhod_tab, text="Left Boundary:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            rhod_left_var = tk.IntVar(value=0)
            rhod_left_slider = ttk.Scale(
                rhod_tab,
                from_=0,
                to=len(self.time_data)-1,
                orient=tk.HORIZONTAL,
                length=200,
                variable=rhod_left_var
            )
            rhod_left_slider.grid(row=1, column=1, padx=5, pady=5)
            
            ttk.Label(rhod_tab, text="Right Boundary:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
            rhod_right_var = tk.IntVar(value=100)
            rhod_right_slider = ttk.Scale(
                rhod_tab,
                from_=0,
                to=len(self.time_data)-1,
                orient=tk.HORIZONTAL,
                length=200,
                variable=rhod_right_var
            )
            rhod_right_slider.grid(row=2, column=1, padx=5, pady=5)
            
            # Update button
            ttk.Button(
                rhod_tab,
                text="Update Boundaries",
                command=lambda: self.update_boundaries("rhod", int(rhod_peak_var.get())-1, rhod_left_var.get(), rhod_right_var.get())
            ).grid(row=3, column=0, columnspan=2, pady=10)
            
            # Function to update sliders when peak selection changes
            def update_rhod_sliders(*args):
                peak_idx = int(rhod_peak_var.get()) - 1
                if 0 <= peak_idx < len(self.rhod_peak_properties[reading_key]):
                    props = self.rhod_peak_properties[reading_key][peak_idx]
                    rhod_left_var.set(props['left_base'])
                    rhod_right_var.set(props['right_base'])
            
            rhod_peak_combo.bind("<<ComboboxSelected>>", update_rhod_sliders)
            update_rhod_sliders()  # Initialize with first peak
        else:
            ttk.Label(rhod_tab, text="No Rhod peaks detected").pack(pady=20)
        
        # Close button
        ttk.Button(
            adjustment_window,
            text="Done",
            command=adjustment_window.destroy
        ).pack(pady=10)
    
    def update_boundaries(self, data_type, peak_idx, left_idx, right_idx):
        reading = int(self.reading_var.get())
        reading_key = f'#{reading}'
        
        # Validate indices
        if left_idx >= right_idx:
            messagebox.showerror("Error", "Left boundary must be less than right boundary")
            return
        
        # Update appropriate peak properties
        if data_type == "fret" and reading_key in self.fret_peak_properties:
            if 0 <= peak_idx < len(self.fret_peak_properties[reading_key]):
                self.fret_peak_properties[reading_key][peak_idx]['left_base'] = left_idx
                self.fret_peak_properties[reading_key][peak_idx]['right_base'] = right_idx
                self.update_plot()
                messagebox.showinfo("Success", f"Updated boundaries for FRET peak #{peak_idx+1}")
        
        elif data_type == "rhod" and reading_key in self.rhod_peak_properties:
            if 0 <= peak_idx < len(self.rhod_peak_properties[reading_key]):
                self.rhod_peak_properties[reading_key][peak_idx]['left_base'] = left_idx
                self.rhod_peak_properties[reading_key][peak_idx]['right_base'] = right_idx
                self.update_plot()
                messagebox.showinfo("Success", f"Updated boundaries for Rhod peak #{peak_idx+1}")

# Create and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = CaMKIIAnalyzer(root)
    root.mainloop()
    
