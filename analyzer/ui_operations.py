import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from pathlib import Path
import sys
from .image_operations import load_ppm_image

class ColorDialog:
    """Dialog for entering color name"""
    def __init__(self, parent, x, y, w, h, canvas, mouse_x, mouse_y):
        self.top = tk.Toplevel(parent)
        self.top.title("Enter Color Name")
        self.top.geometry("300x150")
        self.top.transient(parent)
        self.top.grab_set()
        
        self.result = None
        
        # Position dialog at the mouse release position
        # Convert canvas coordinates to screen coordinates
        canvas_x = canvas.canvasx(mouse_x)
        canvas_y = canvas.canvasy(mouse_y)
        
        # Get screen coordinates of the mouse position
        screen_x = canvas.winfo_rootx() + int(canvas_x)
        screen_y = canvas.winfo_rooty() + int(canvas_y)
        
        # Position dialog near the mouse, but ensure it stays on screen
        screen_width = self.top.winfo_screenwidth()
        screen_height = self.top.winfo_screenheight()
        
        # Adjust position if dialog would go off screen
        dialog_width = 300
        dialog_height = 150
        
        final_x = screen_x
        final_y = screen_y
        
        # Ensure dialog stays on screen
        if final_x + dialog_width > screen_width:
            final_x = screen_width - dialog_width - 20
        if final_y + dialog_height > screen_height:
            final_y = screen_height - dialog_height - 20
        if final_x < 0:
            final_x = 20
        if final_y < 0:
            final_y = 20
            
        self.top.geometry(f"+{final_x}+{final_y}")
        
        # Selection info
        info_label = tk.Label(self.top, text=f"Selection: ({x}, {y}) {w}x{h}")
        info_label.pack(pady=10)
        
        # Color entry
        color_frame = tk.Frame(self.top)
        color_frame.pack(pady=10)
        
        tk.Label(color_frame, text="Color:").pack(side=tk.LEFT)
        self.color_var = tk.StringVar()
        self.color_entry = tk.Entry(color_frame, textvariable=self.color_var, width=20)
        self.color_entry.pack(side=tk.LEFT, padx=5)
        self.color_entry.focus_set()
        
        # Buttons
        button_frame = tk.Frame(self.top)
        button_frame.pack(pady=10)
        
        self.ok_button = tk.Button(button_frame, text="OK", command=self.ok, state=tk.DISABLED)
        self.ok_button.pack(side=tk.LEFT, padx=5)
        
        cancel_button = tk.Button(button_frame, text="Cancel", command=self.cancel)
        cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Bind entry validation
        self.color_var.trace('w', self.validate_entry)
        self.color_entry.bind('<Return>', lambda e: self.ok() if self.ok_button['state'] == tk.NORMAL else None)
    
    def validate_entry(self, *args):
        """Enable OK button only when color is entered"""
        if self.color_var.get().strip():
            self.ok_button.config(state=tk.NORMAL)
        else:
            self.ok_button.config(state=tk.DISABLED)
    
    def ok(self):
        """OK button handler"""
        color = self.color_var.get().strip()
        if color:
            self.result = color
            self.top.destroy()
    
    def cancel(self):
        """Cancel button handler"""
        self.top.destroy()

def setup_main_window(title):
    """Setup the main Tkinter window"""
    root = tk.Tk()
    root.title(title)
    return root

def calculate_initial_zoom(original_img, screen_width, screen_height, margin=100):
    """Calculate zoom level to fit image on screen"""
    max_display_width = screen_width - margin
    max_display_height = screen_height - margin
    
    width_zoom = max_display_width / original_img.width
    height_zoom = max_display_height / original_img.height
    
    return min(width_zoom, height_zoom, 1.0)

def create_scrollable_canvas(parent, width, height):
    """Create canvas with scrollbars"""
    frame = tk.Frame(parent)
    frame.pack(fill=tk.BOTH, expand=True)
    
    v_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
    h_scrollbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
    
    canvas = tk.Canvas(frame, width=width, height=height,
                      yscrollcommand=v_scrollbar.set,
                      xscrollcommand=h_scrollbar.set)
    
    v_scrollbar.config(command=canvas.yview)
    h_scrollbar.config(command=canvas.xview)
    
    v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    return canvas

def get_zoomed_image(original_img, zoom, zoom_cache):
    """Get zoomed image from cache or create it"""
    cache_key = round(zoom, 2)
    
    if cache_key not in zoom_cache:
        new_width = int(original_img.width * cache_key)
        new_height = int(original_img.height * cache_key)
        
        # Use faster resampling for large images
        if new_width > 2000 or new_height > 2000:
            resample_method = Image.Resampling.NEAREST
        else:
            resample_method = Image.Resampling.LANCZOS
        
        zoomed_img = original_img.resize((new_width, new_height), resample_method)
        zoom_cache[cache_key] = zoomed_img
        
        # Limit cache size
        if len(zoom_cache) > 10:
            oldest_key = min(zoom_cache.keys())
            del zoom_cache[oldest_key]
    
    return zoom_cache[cache_key]

def update_canvas_image(canvas, original_img, zoom_level, zoom_cache):
    """Update the canvas with current zoomed image"""
    zoomed_img = get_zoomed_image(original_img, zoom_level, zoom_cache)
    current_photo = ImageTk.PhotoImage(zoomed_img)
    
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=current_photo)
    canvas.config(scrollregion=(0, 0, zoomed_img.width, zoomed_img.height))
    canvas.image = current_photo
    return current_photo

def canvas_to_image_coords(canvas_x, canvas_y, zoom_level):
    """Convert canvas coordinates to original image coordinates"""
    image_x = int(canvas_x / zoom_level)
    image_y = int(canvas_y / zoom_level)
    return image_x, image_y

def handle_zoom(event, zoom_level, zoom_factor, min_zoom, max_zoom, update_callback):
    """Handle zoom with mouse wheel"""
    if event.delta > 0:  # Zoom in
        new_zoom = zoom_level * zoom_factor
    else:  # Zoom out
        new_zoom = zoom_level / zoom_factor
    
    # Clamp zoom level
    new_zoom = max(min_zoom, min(max_zoom, new_zoom))
    
    if new_zoom != zoom_level:
        update_callback(new_zoom)
    
    return new_zoom

def handle_scroll(event, canvas, is_horizontal=False):
    """Handle scrolling with mouse wheel"""
    delta = get_mouse_wheel_delta(event)
    
    # Convert delta to scroll units (typically -1, 0, or 1)
    scroll_units = -1 if delta > 0 else 1 if delta < 0 else 0
    
    if is_horizontal:
        canvas.xview_scroll(scroll_units, "units")
    else:
        canvas.yview_scroll(scroll_units, "units")

def get_mouse_wheel_delta(event):
    """Get normalized mouse wheel delta across platforms"""
    if hasattr(event, 'delta'):
        return event.delta  # Windows/Mac - keep original sign
    elif event.num == 4:
        return 120  # Linux scroll up
    elif event.num == 5:
        return -120  # Linux scroll down
    return 0

def is_modifier_pressed(event, mask):
    """Check if modifier key is pressed"""
    return (event.state & mask) != 0

def handle_mouse_wheel(event, zoom_level_ref, zoom_factor, min_zoom, max_zoom, canvas, update_callback):
    """Handle mouse wheel events for scrolling and zooming"""
    delta = get_mouse_wheel_delta(event)
    if delta == 0:
        return
    
    ctrl_pressed = is_modifier_pressed(event, 0x0004)
    alt_pressed = is_modifier_pressed(event, 0x0008)
    shift_pressed = is_modifier_pressed(event, 0x0001)
    
    if ctrl_pressed:
        # Zoom with Ctrl
        if delta > 0:  # Zoom in
            new_zoom = zoom_level_ref[0] * zoom_factor
        else:  # Zoom out
            new_zoom = zoom_level_ref[0] / zoom_factor
        
        # Clamp zoom level
        new_zoom = max(min_zoom, min(max_zoom, new_zoom))
        
        if new_zoom != zoom_level_ref[0]:
            zoom_level_ref[0] = new_zoom
            update_callback(new_zoom)
            
        # Adjust scroll position to zoom towards mouse position
        zoom_center_x = canvas.canvasx(event.x)
        zoom_center_y = canvas.canvasy(event.y)
        if zoom_center_x is not None and zoom_center_y is not None:
            canvas.scan_mark(int(zoom_center_x), int(zoom_center_y))
            canvas.scan_dragto(int(zoom_center_x), int(zoom_center_y), gain=1)
    
    elif alt_pressed or shift_pressed:
        # Horizontal scroll with Alt or Shift
        handle_scroll(event, canvas, is_horizontal=True)
    else:
        # Vertical scroll (no modifiers)
        handle_scroll(event, canvas, is_horizontal=False)

def setup_selection_handlers(canvas, zoom_level, original_img, image_path):
    """Setup mouse handlers for selection"""
    start_x, start_y = None, None
    rect_id = None
    
    def on_button_press(event):
        nonlocal start_x, start_y, rect_id
        start_x = canvas.canvasx(event.x)
        start_y = canvas.canvasy(event.y)
        rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=2)
    
    def on_motion(event):
        nonlocal rect_id
        if rect_id:
            current_x = canvas.canvasx(event.x)
            current_y = canvas.canvasy(event.y)
            canvas.coords(rect_id, start_x, start_y, current_x, current_y)
    
    def on_button_release(event):
        nonlocal start_x, start_y, rect_id
        if not start_x or not start_y:
            return
        
        # Convert canvas coordinates to image coordinates
        end_x = canvas.canvasx(event.x)
        end_y = canvas.canvasy(event.y)
        
        img_x1, img_y1 = canvas_to_image_coords(start_x, start_y, zoom_level)
        img_x2, img_y2 = canvas_to_image_coords(end_x, end_y, zoom_level)
        
        # Ensure coordinates are within image bounds
        x = min(max(0, img_x1), original_img.width - 1)
        y = min(max(0, img_y1), original_img.height - 1)
        w = abs(img_x2 - img_x1)
        h = abs(img_y2 - img_y1)
        
        # Skip if selection is too small
        if w < 5 or h < 5:
            canvas.delete(rect_id)
            return
        
        # Show color selection dialog at the mouse release position
        show_color_dialog(x, y, w, h, canvas, rect_id, original_img, image_path, end_x, end_y)
    
    canvas.bind("<ButtonPress-1>", on_button_press)
    canvas.bind("<B1-Motion>", on_motion)
    canvas.bind("<ButtonRelease-1>", on_button_release)

def show_color_dialog(x, y, w, h, canvas, rect_id, original_img, image_path, mouse_x, mouse_y):
    """Show dialog to enter color name at mouse position"""
    dialog = ColorDialog(canvas, x, y, w, h, canvas, mouse_x, mouse_y)
    canvas.wait_window(dialog.top)
    
    if dialog.result:
        save_sample(x, y, w, h, dialog.result, original_img, image_path)
    
    canvas.delete(rect_id)

def save_sample(x, y, w, h, color, original_img, image_path):
    """Save the selected sample"""
    try:
        # Extract subimage
        subimage = original_img.crop((x, y, x + w, y + h))
        
        # Create samples directory
        input_path = Path(image_path)
        samples_dir = input_path.parent / (input_path.stem + '.samples')
        samples_dir.mkdir(exist_ok=True)
        
        # Save sample
        sample_filename = f"{x}_{y}_{w}_{h}.{color}"
        sample_path = samples_dir / sample_filename
        subimage.save(sample_path, format='PPM')
        
        print(f"Sample saved to: {sample_path}")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save sample: {e}")

def setup_mouse_wheel_bindings(canvas, zoom_level, zoom_factor, min_zoom, max_zoom, update_callback):
    """Bind mouse wheel events for zooming and scrolling"""
    # Use a list to allow modification by reference
    zoom_level_ref = [zoom_level]
    
    def mouse_wheel_handler(event):
        handle_mouse_wheel(event, zoom_level_ref, zoom_factor, min_zoom, max_zoom, canvas, update_callback)
    
    # Bind to canvas
    canvas.bind("<MouseWheel>", mouse_wheel_handler)
    canvas.bind("<Button-4>", mouse_wheel_handler)
    canvas.bind("<Button-5>", mouse_wheel_handler)
    
    # Also bind to parent to ensure events are captured when not over image
    parent = canvas.master.master  # Get the main frame
    parent.bind("<MouseWheel>", mouse_wheel_handler)
    parent.bind("<Button-4>", mouse_wheel_handler)
    parent.bind("<Button-5>", mouse_wheel_handler)
    
    return zoom_level_ref

def sampleselector(image_path):
    """Interactive sample selection tool"""
    try:
        # Disable decompression bomb protection for large images
        Image.MAX_IMAGE_PIXELS = None
        
        # Load the PPM image
        original_img = load_ppm_image(image_path, require_ppm=True)
        
        # Create main window
        root = setup_main_window("Sample Selector - Drag to select, Ctrl+MouseWheel to zoom, MouseWheel to scroll")
        
        # Calculate initial zoom and canvas size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        zoom_level = calculate_initial_zoom(original_img, screen_width, screen_height)
        canvas_width = int(original_img.width * zoom_level)
        canvas_height = int(original_img.height * zoom_level)
        
        # Create canvas
        canvas = create_scrollable_canvas(root, canvas_width, canvas_height)
        
        # Setup zoom state and cache
        min_zoom = 0.1
        max_zoom = 10.0
        zoom_factor = 1.2
        zoom_cache = {}
        
        def update_callback(new_zoom):
            nonlocal zoom_level
            zoom_level = new_zoom
            update_canvas_image(canvas, original_img, zoom_level, zoom_cache)
        
        # Setup event handlers
        setup_selection_handlers(canvas, zoom_level, original_img, image_path)
        zoom_level_ref = setup_mouse_wheel_bindings(canvas, zoom_level, zoom_factor, min_zoom, max_zoom, update_callback)
        
        # Update the update_callback to modify the reference
        def update_callback(new_zoom):
            zoom_level_ref[0] = new_zoom
            update_canvas_image(canvas, original_img, new_zoom, zoom_cache)
        
        # Initial image display
        update_canvas_image(canvas, original_img, zoom_level, zoom_cache)
        
        # Handle window close
        def on_closing():
            root.quit()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        print("Sample Selector started. Drag to select regions, mouse wheel to zoom, close window to exit.")
        root.mainloop()
        
    except Exception as e:
        print(f"Error in sampleselector: {e}")
        sys.exit(1)
