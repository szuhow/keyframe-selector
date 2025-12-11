import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import networkx as nx

# ==================== FUNKCJE POMOCNICZE ====================

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_data_from_directory(base_dir):
    """Ładuje maski i oryginalne obrazy z visualizations."""
    pred_dir = os.path.join(base_dir, 'predictions')
    viz_dir = os.path.join(base_dir, 'visualizations')
    
    masks = []
    originals = []
    filenames = []
    
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('_pred.png')], key=natural_sort_key)
    print(f"Znaleziono {len(pred_files)} plików")
    
    for filename in pred_files:
        # Maska - czytaj jako kolor i bierz max z kanałów
        mask_path = os.path.join(pred_dir, filename)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if mask_img is None:
            continue
        binary_mask = (np.max(mask_img, axis=2) > 10).astype(np.uint8)
        
        # Oryginalny obraz z visualizations - spróbuj różne warianty nazw
        possible_names = [
            filename.replace('_pred.png', '.png'),      # bez _pred
            filename.replace('_pred.png', '_viz.png'),  # z _viz
        ]
        
        orig_gray = None
        for orig_name in possible_names:
            orig_path = os.path.join(viz_dir, orig_name)
            if os.path.exists(orig_path):
                viz_img = cv2.imread(orig_path, cv2.IMREAD_COLOR)
                if viz_img is not None:
                    h, w = viz_img.shape[:2]
                    # Sprawdź czy to tryptyk (3 obrazy obok siebie)
                    if w > h * 2:  # Szeroki obraz = tryptyk
                        orig = viz_img[:, :w//3]
                    else:
                        orig = viz_img
                    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
                    # Przeskaluj do rozmiaru maski
                    if orig_gray.shape != binary_mask.shape:
                        orig_gray = cv2.resize(orig_gray, (binary_mask.shape[1], binary_mask.shape[0]))
                    break
        
        if orig_gray is None:
            orig_gray = np.zeros_like(binary_mask, dtype=np.uint8)
        
        masks.append(binary_mask)
        originals.append(orig_gray)
        filenames.append(filename)
    
    return masks, originals, filenames

# ==================== METODA 1: MASKED OPTICAL FLOW ====================

class OpticalFlowKeyframeSelector:
    """Selektor oparty na Optical Flow z maską naczyń."""
    
    def __init__(self, masks, originals):
        self.masks = np.array(masks)
        self.originals = np.array(originals)
        self.n_frames = len(masks)
        
    def compute_masked_optical_flow(self, n_points=80):
        """Oblicza ruch naczyń przez śledzenie punktów kontrolnych (bidirectional)."""
        
        # Znajdź klatkę z max wypełnieniem jako źródło punktów
        fill_scores = [np.sum(m) for m in self.masks]
        init_frame = np.argmax(fill_scores)
        
        # Znajdź punkty kontrolne na naczyniu
        mask_dilated = cv2.dilate(self.masks[init_frame], np.ones((5,5), np.uint8), iterations=1)
        corners = cv2.goodFeaturesToTrack(
            self.originals[init_frame],
            maxCorners=n_points,
            qualityLevel=0.01,
            minDistance=10,
            mask=mask_dilated,
            blockSize=7
        )
        
        if corners is None or len(corners) == 0:
            self.motion_scores = np.zeros(self.n_frames)
            return self.motion_scores
        
        init_points = corners.reshape(-1, 2)
        n_pts = len(init_points)
        
        # Pozycje punktów dla każdej klatki
        frame_positions = {init_frame: init_points.copy()}
        
        # Tracking wstecz
        current_points = init_points.copy()
        active = np.ones(n_pts, dtype=bool)
        
        for frame_idx in range(init_frame, 0, -1):
            curr_gray = self.originals[frame_idx]
            prev_gray = self.originals[frame_idx - 1]
            prev_mask = self.masks[frame_idx - 1]
            
            flow = cv2.calcOpticalFlowFarneback(
                curr_gray, prev_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            new_points = current_points.copy()
            for i in range(n_pts):
                if not active[i]:
                    continue
                x, y = current_points[i]
                x_int, y_int = int(round(x)), int(round(y))
                if 0 <= y_int < flow.shape[0] and 0 <= x_int < flow.shape[1]:
                    dx, dy = flow[y_int, x_int]
                    new_x, new_y = x + dx, y + dy
                    ny, nx = int(round(new_y)), int(round(new_x))
                    if 0 <= ny < prev_mask.shape[0] and 0 <= nx < prev_mask.shape[1]:
                        y1, y2 = max(0, ny-5), min(prev_mask.shape[0], ny+6)
                        x1, x2 = max(0, nx-5), min(prev_mask.shape[1], nx+6)
                        if np.any(prev_mask[y1:y2, x1:x2]):
                            new_points[i] = [new_x, new_y]
                        else:
                            active[i] = False
                    else:
                        active[i] = False
                else:
                    active[i] = False
            
            current_points = new_points
            frame_positions[frame_idx - 1] = current_points.copy()
        
        # Tracking wprzód
        current_points = init_points.copy()
        active = np.ones(n_pts, dtype=bool)
        
        for frame_idx in range(init_frame, self.n_frames - 1):
            prev_gray = self.originals[frame_idx]
            curr_gray = self.originals[frame_idx + 1]
            curr_mask = self.masks[frame_idx + 1]
            
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            new_points = current_points.copy()
            for i in range(n_pts):
                if not active[i]:
                    continue
                x, y = current_points[i]
                x_int, y_int = int(round(x)), int(round(y))
                if 0 <= y_int < flow.shape[0] and 0 <= x_int < flow.shape[1]:
                    dx, dy = flow[y_int, x_int]
                    new_x, new_y = x + dx, y + dy
                    ny, nx = int(round(new_y)), int(round(new_x))
                    if 0 <= ny < curr_mask.shape[0] and 0 <= nx < curr_mask.shape[1]:
                        y1, y2 = max(0, ny-5), min(curr_mask.shape[0], ny+6)
                        x1, x2 = max(0, nx-5), min(curr_mask.shape[1], nx+6)
                        if np.any(curr_mask[y1:y2, x1:x2]):
                            new_points[i] = [new_x, new_y]
                        else:
                            active[i] = False
                    else:
                        active[i] = False
                else:
                    active[i] = False
            
            current_points = new_points
            frame_positions[frame_idx + 1] = current_points.copy()
        
        # Oblicz motion score dla każdej klatki (średnie przesunięcie punktów)
        motion_scores = np.zeros(self.n_frames)
        for frame_idx in range(1, self.n_frames):
            if frame_idx in frame_positions and (frame_idx - 1) in frame_positions:
                prev_pts = frame_positions[frame_idx - 1]
                curr_pts = frame_positions[frame_idx]
                displacements = np.sqrt(np.sum((curr_pts - prev_pts) ** 2, axis=1))
                motion_scores[frame_idx] = np.mean(displacements)
        
        self.motion_scores = motion_scores
        self.control_point_positions = frame_positions
        return self.motion_scores
    
    def compute_fill_scores(self):
        self.fill_scores = np.array([np.sum(m) / m.size for m in self.masks])
        return self.fill_scores
    
    def compute_thickness_scores(self):
        self.thickness_scores = []
        for mask in self.masks:
            if np.sum(mask) == 0:
                self.thickness_scores.append(0)
                continue
            dist = ndimage.distance_transform_edt(mask.astype(bool))
            skeleton = morphology.skeletonize(mask.astype(bool))
            if np.sum(skeleton) > 0:
                thickness = np.median(dist[skeleton]) * 2
            else:
                thickness = 0
            self.thickness_scores.append(thickness)
        self.thickness_scores = np.array(self.thickness_scores)
        return self.thickness_scores
    
    def normalize(self, arr):
        arr = np.array(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.ones_like(arr) * 0.5
    
    def compute_cardiac_phase(self):
        """Oblicza fazę serca przez analizę ruchu w obszarze naczyń + fill_score."""
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks
        
        h, w = self.originals[0].shape
        
        # Sygnały do analizy
        divergence_vessel = []  # divergence tylko w obszarze naczyń
        fill_signal = []        # wypełnienie maski
        
        for frame_idx in range(self.n_frames - 1):
            prev_gray = self.originals[frame_idx]
            curr_gray = self.originals[frame_idx + 1]
            mask = self.masks[frame_idx]
            
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Divergence w obszarze naczyń (ekspansja/kontrakcja)
            du_dx = np.gradient(flow[..., 0], axis=1)
            dv_dy = np.gradient(flow[..., 1], axis=0)
            div = du_dx + dv_dy
            
            if np.sum(mask) > 100:
                div_vessel = np.mean(div[mask > 0])
            else:
                div_vessel = 0
            
            divergence_vessel.append(div_vessel)
            fill_signal.append(np.sum(mask))
        
        divergence_vessel.append(0)
        fill_signal.append(fill_signal[-1] if fill_signal else 0)
        
        divergence_vessel = np.array(divergence_vessel)
        fill_signal = np.array(fill_signal, dtype=float)
        
        # Normalizuj fill_signal
        if fill_signal.max() > fill_signal.min():
            fill_norm = (fill_signal - fill_signal.min()) / (fill_signal.max() - fill_signal.min())
        else:
            fill_norm = np.ones(self.n_frames) * 0.5
        
        # Lekkie wygładzenie
        div_smooth = gaussian_filter1d(divergence_vessel, sigma=1)
        fill_smooth = gaussian_filter1d(fill_norm, sigma=1)
        
        # End-diastolic = lokalne maksimum fill (maksymalne wypełnienie kontrastu)
        # Użyj niższych progów żeby znaleźć wszystkie cykle
        
        # Znajdź lokalne maksima fill (niski próg prominence)
        fill_peaks, _ = find_peaks(fill_smooth, distance=3, prominence=0.02)
        
        # Znajdź lokalne maksima divergence
        div_peaks, _ = find_peaks(div_smooth, distance=3, prominence=0.001)
        
        # Połącz - preferuj punkty gdzie oba sygnały mają maksimum
        end_diastolic_frames = []
        
        # Najpierw fill peaks
        for peak in fill_peaks:
            end_diastolic_frames.append(peak)
        
        # Dodaj div peaks jeśli są blisko fill peaks lub jeśli fill peaks puste
        for peak in div_peaks:
            if div_smooth[peak] > 0:  # tylko ekspansja
                # Sprawdź czy nie ma już bliskiego end_diastolic
                if not any(abs(peak - ed) < 3 for ed in end_diastolic_frames):
                    end_diastolic_frames.append(peak)
        
        end_diastolic_frames = sorted(set(end_diastolic_frames))
        
        # Fallback: jeśli brak peaks, użyj klatek z wysokim fill
        if not end_diastolic_frames:
            threshold = np.percentile(fill_smooth, 75)
            end_diastolic_frames = list(np.where(fill_smooth >= threshold)[0])
        
        # Przypisz score: gaussowski spadek od end-diastolic frames
        phase_scores = np.zeros(self.n_frames)
        sigma = 2  # węższe okno żeby pokazać wszystkie cykle
        
        for ed_frame in end_diastolic_frames:
            for i in range(self.n_frames):
                dist = abs(i - ed_frame)
                phase_scores[i] = max(phase_scores[i], np.exp(-dist**2 / (2 * sigma**2)))
        
        # Normalizuj
        if np.max(phase_scores) > 0:
            phase_scores = phase_scores / np.max(phase_scores)
        else:
            phase_scores = fill_smooth  # fallback do fill
        
        self.phase_scores = phase_scores
        self.radial_flow = div_smooth  # zachowaj dla kompatybilności z wizualizacją
        self.fill_smooth = fill_smooth
        self.end_diastolic_frames = end_diastolic_frames
        return self.phase_scores
    
    def compute_diastole_mask(self):
        """Oblicza maskę rozkurczu przez analizę ruchu radialnego naczyń."""
        from scipy.ndimage import gaussian_filter1d
        
        h, w = self.originals[0].shape
        
        # Znajdź centrum masy naczyń
        all_pts = []
        for mask in self.masks:
            pts = np.argwhere(mask > 0)
            if len(pts) > 0:
                all_pts.extend(pts)
        if all_pts:
            cy, cx = np.mean(all_pts, axis=0)
        else:
            cy, cx = h // 2, w // 2
        
        # CLAHE dla lepszego optical flow
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        outward_ratio = []
        for frame_idx in range(self.n_frames - 1):
            prev_eq = clahe.apply(self.originals[frame_idx])
            curr_eq = clahe.apply(self.originals[frame_idx + 1])
            prev_eq = cv2.GaussianBlur(prev_eq, (5, 5), 0)
            curr_eq = cv2.GaussianBlur(curr_eq, (5, 5), 0)
            
            flow = cv2.calcOpticalFlowFarneback(
                prev_eq, curr_eq, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            mask = self.masks[frame_idx]
            outward_cnt = 0
            inward_cnt = 0
            vessel_pts = np.argwhere(mask > 0)
            for pt in vessel_pts[::5]:
                y, x = pt
                dx, dy = flow[y, x]
                rx, ry = x - cx, y - cy
                r_len = np.sqrt(rx**2 + ry**2)
                if r_len > 10:
                    radial = (dx * rx + dy * ry) / r_len
                    if radial > 0.3:
                        outward_cnt += 1
                    elif radial < -0.3:
                        inward_cnt += 1
            
            total = outward_cnt + inward_cnt
            if total > 50:
                outward_ratio.append(outward_cnt / total)
            else:
                outward_ratio.append(0.5)
        
        outward_ratio.append(0.5)
        outward_ratio = np.array(outward_ratio)
        
        # Rozkurcz = outward_ratio > 0.5
        self.is_diastole = outward_ratio > 0.5
        self.outward_ratio = outward_ratio
        return self.is_diastole
    
    def select_best_frame(self, w_fill=0.4, w_thick=0.4, w_stability=0.1, w_phase=0.1):
        self.compute_fill_scores()
        self.compute_thickness_scores()
        self.compute_masked_optical_flow()
        self.compute_cardiac_phase()
        self.compute_diastole_mask()
        
        norm_fill = self.normalize(self.fill_scores)
        norm_thick = self.normalize(self.thickness_scores)
        norm_stability = 1.0 - self.normalize(self.motion_scores)
        norm_phase = self.phase_scores
        
        scores = (w_fill * norm_fill + 
                  w_thick * norm_thick + 
                  w_stability * norm_stability + 
                  w_phase * norm_phase)
        
        # Próg: odrzuć klatki z wypełnieniem < 20% maksymalnego
        max_fill = np.max(self.fill_scores)
        for i in range(len(scores)):
            if self.fill_scores[i] < max_fill * 0.2:
                scores[i] = 0
        
        # PREFERUJ ROZKURCZ: penalizuj klatki w skurczu
        for i in range(len(scores)):
            if not self.is_diastole[i]:
                scores[i] *= 0.5  # 50% kary za skurcz
        
        best_idx = np.argmax(scores)
        
        n_diastole = np.sum(self.is_diastole)
        print(f"  Klatki w rozkurczu: {n_diastole}/{self.n_frames}")
        print(f"  End-diastolic frames: {self.end_diastolic_frames}")
        
        return {
            'best_frame': best_idx,
            'scores': scores,
            'metrics': {
                'fill': norm_fill,
                'thickness': norm_thick,
                'stability': norm_stability,
                'phase': norm_phase
            }
        }

# ==================== METODA 2: GRAPH MATCHING ====================

class GraphMatchingKeyframeSelector:
    """Selektor oparty na dopasowaniu grafów topologii naczyń."""
    
    def __init__(self, masks):
        self.masks = np.array(masks)
        self.n_frames = len(masks)
        self.graphs = []
        
    def build_vessel_graph(self, mask):
        """Buduje graf z topologii naczyń."""
        if np.sum(mask) == 0:
            return nx.Graph()
        
        # Odszumianie przed szkieletyzacją
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        skeleton = morphology.skeletonize(cleaned.astype(bool))
        if np.sum(skeleton) == 0:
            return nx.Graph()
        
        # Znajdź punkty charakterystyczne
        kernel_n = np.array([[1,1,1],[1,0,1],[1,1,1]])
        neighbors = ndimage.convolve(skeleton.astype(int), kernel_n, mode='constant')
        neighbors = neighbors * skeleton
        
        endpoints = np.argwhere((neighbors == 1) & skeleton)
        bifurcations = np.argwhere((neighbors >= 3) & skeleton)
        
        G = nx.Graph()
        nodes = []
        
        for pt in endpoints:
            nodes.append(('E', tuple(pt)))
        for pt in bifurcations:
            nodes.append(('B', tuple(pt)))
        
        for i, (ntype, pos) in enumerate(nodes):
            G.add_node(i, pos=pos, type=ntype)
        
        # Krawędzie - połączenie bliskich węzłów
        if len(nodes) > 1:
            positions = np.array([n[1] for n in nodes])
            dists = cdist(positions, positions)
            
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    if dists[i,j] < 100:
                        G.add_edge(i, j, weight=dists[i,j])
        
        return G
    
    def compute_graph_similarity(self, G1, G2):
        """Oblicza podobieństwo dwóch grafów."""
        if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
            return 0.0
        
        n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
        e1, e2 = G1.number_of_edges(), G2.number_of_edges()
        
        node_sim = 1.0 - abs(n1 - n2) / max(n1, n2, 1)
        edge_sim = 1.0 - abs(e1 - e2) / max(e1, e2, 1)
        
        deg1 = sorted([d for _, d in G1.degree()])
        deg2 = sorted([d for _, d in G2.degree()])
        
        max_len = max(len(deg1), len(deg2))
        deg1 = deg1 + [0] * (max_len - len(deg1))
        deg2 = deg2 + [0] * (max_len - len(deg2))
        
        deg_diff = np.sum(np.abs(np.array(deg1) - np.array(deg2)))
        deg_sim = 1.0 / (1.0 + deg_diff)
        
        return 0.4 * node_sim + 0.3 * edge_sim + 0.3 * deg_sim
    
    def compute_stability_scores(self):
        """Stabilność topologii między klatkami."""
        self.graphs = [self.build_vessel_graph(m) for m in self.masks]
        
        stability = [1.0]
        for i in range(1, self.n_frames):
            sim = self.compute_graph_similarity(self.graphs[i-1], self.graphs[i])
            stability.append(sim)
        
        self.stability_scores = np.array(stability)
        return self.stability_scores
    
    def compute_complexity_scores(self):
        """Złożoność grafu."""
        if not self.graphs:
            self.graphs = [self.build_vessel_graph(m) for m in self.masks]
        
        complexity = []
        for G in self.graphs:
            n = G.number_of_nodes()
            e = G.number_of_edges()
            complexity.append(n + e)
        
        self.complexity_scores = np.array(complexity, dtype=float)
        return self.complexity_scores
    
    def compute_fill_scores(self):
        self.fill_scores = np.array([np.sum(m) / m.size for m in self.masks])
        return self.fill_scores
    
    def normalize(self, arr):
        arr = np.array(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.ones_like(arr) * 0.5
    
    def select_best_frame(self, w_stability=0.3, w_complexity=0.3, w_fill=0.4):
        self.compute_stability_scores()
        self.compute_complexity_scores()
        self.compute_fill_scores()
        
        norm_stability = self.normalize(self.stability_scores)
        norm_complexity = self.normalize(self.complexity_scores)
        norm_fill = self.normalize(self.fill_scores)
        
        scores = w_stability * norm_stability + w_complexity * norm_complexity + w_fill * norm_fill
        
        # Próg: odrzuć klatki z < 3 węzłami lub niskim wypełnieniem
        max_fill = np.max(self.fill_scores)
        for i in range(len(scores)):
            if self.graphs[i].number_of_nodes() < 3 or self.fill_scores[i] < max_fill * 0.2:
                scores[i] = 0
        
        best_idx = np.argmax(scores)
        
        return {
            'best_frame': best_idx,
            'scores': scores,
            'metrics': {
                'stability': norm_stability,
                'complexity': norm_complexity,
                'fill': norm_fill
            }
        }

# ==================== DIAGNOSTYKA OPTICAL FLOW ====================

def visualize_optical_flow_debug(selector, result):
    """Szczegółowa diagnostyka Optical Flow."""
    n = selector.n_frames
    
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3)
    
    # 1. Surowe wartości ruchu (nie znormalizowane)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(range(n), selector.motion_scores, 'r-o', markersize=3, label='Motion (raw)')
    ax1.set_title('Optical Flow - Surowy ruch (magnitude)')
    ax1.set_xlabel('Klatka')
    ax1.set_ylabel('Średni ruch [px]')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(np.mean(selector.motion_scores), color='gray', linestyle='--', label=f'Mean: {np.mean(selector.motion_scores):.2f}')
    ax1.legend()
    
    # 2. Surowe wypełnienie
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.plot(range(n), selector.fill_scores * 100, 'b-o', markersize=3, label='Fill %')
    ax2.set_title('Wypełnienie maski (%)')
    ax2.set_xlabel('Klatka')
    ax2.set_ylabel('Wypełnienie [%]')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Surowa grubość
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(range(n), selector.thickness_scores, 'g-o', markersize=3, label='Thickness [px]')
    ax3.set_title('Grubość naczyń (mediana)')
    ax3.set_xlabel('Klatka')
    ax3.set_ylabel('Grubość [px]')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Stabilność = 1 - normalized motion
    ax4 = fig.add_subplot(gs[1, 2:])
    stability = result['metrics']['stability']
    ax4.bar(range(n), stability, color='green', alpha=0.7, edgecolor='black')
    ax4.set_title('Stabilność (1 - norm_motion)')
    ax4.set_xlabel('Klatka')
    ax4.set_ylabel('Stabilność')
    ax4.axhline(0.5, color='red', linestyle='--')
    ax4.grid(True, alpha=0.3)
    
    # 5. Faza serca - divergence + fill
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.plot(range(n), selector.radial_flow, 'purple', linewidth=1.5, label='Divergence (naczynia)')
    if hasattr(selector, 'fill_smooth'):
        ax5.plot(range(n), selector.fill_smooth, 'cyan', linewidth=1.5, label='Fill (norm)')
    ax5.axhline(0, color='black', linestyle='--', linewidth=0.5)
    for ed in selector.end_diastolic_frames:
        ax5.axvline(ed, color='red', linestyle='--', alpha=0.7, label='End-diastolic' if ed == selector.end_diastolic_frames[0] else '')
    ax5.set_title('Divergence naczyń + Fill | czerwone = end-diastolic')
    ax5.set_xlabel('Klatka')
    ax5.set_ylabel('Wartość')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Phase score
    ax6 = fig.add_subplot(gs[2, 2:])
    phase = result['metrics']['phase']
    ax6.bar(range(n), phase, color='orange', alpha=0.7, edgecolor='black')
    for ed in selector.end_diastolic_frames:
        ax6.axvline(ed, color='red', linestyle='--', alpha=0.7)
    ax6.set_title('Phase Score (preferencja end-diastolic)')
    ax6.set_xlabel('Klatka')
    ax6.set_ylabel('Phase score')
    ax6.grid(True, alpha=0.3)
    
    # 7-10. Optical flow dla wybranych klatek
    sample_frames = [5, 15, 25, min(n-2, 35)]
    sample_frames = [f for f in sample_frames if f < n-1]
    
    for idx, frame_idx in enumerate(sample_frames[:4]):
        ax = fig.add_subplot(gs[3, idx])
        
        prev_gray = selector.originals[frame_idx]
        curr_gray = selector.originals[frame_idx + 1]
        mask = selector.masks[frame_idx]
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Pokaż magnitude z maską
        masked_mag = mag * mask
        ax.imshow(masked_mag, cmap='hot')
        ax.set_title(f'Flow mag #{frame_idx}→{frame_idx+1}\nMean: {np.mean(masked_mag[mask>0]):.2f}px')
        ax.axis('off')
    
    # Maski dla tych samych klatek
    for idx, frame_idx in enumerate(sample_frames[:4]):
        ax = fig.add_subplot(gs[4, idx])
        ax.imshow(selector.masks[frame_idx], cmap='gray')
        ax.set_title(f'Maska #{frame_idx}\nFill: {selector.fill_scores[frame_idx]*100:.1f}%')
        ax.axis('off')
    
    plt.suptitle('DIAGNOSTYKA: Optical Flow + Cardiac Phase', fontsize=14, fontweight='bold')
    plt.savefig('optical_flow_debug.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==================== DIAGNOSTYKA GRAPH MATCHING ====================

def visualize_graph_debug(selector, result):
    """Szczegółowa diagnostyka Graph Matching."""
    n = selector.n_frames
    
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.3)
    
    # 1. Surowa stabilność grafu
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(range(n), selector.stability_scores, 'b-o', markersize=3, label='Graph Similarity')
    ax1.set_title('Podobieństwo grafu między kolejnymi klatkami')
    ax1.set_xlabel('Klatka')
    ax1.set_ylabel('Similarity [0-1]')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(np.mean(selector.stability_scores), color='gray', linestyle='--')
    ax1.legend()
    
    # 2. Złożoność grafu
    ax2 = fig.add_subplot(gs[0, 2:])
    nodes = [G.number_of_nodes() for G in selector.graphs]
    edges = [G.number_of_edges() for G in selector.graphs]
    ax2.bar(range(n), nodes, alpha=0.7, label='Węzły', color='blue')
    ax2.bar(range(n), edges, alpha=0.5, label='Krawędzie', color='orange', bottom=nodes)
    ax2.set_title('Złożoność grafu (węzły + krawędzie)')
    ax2.set_xlabel('Klatka')
    ax2.set_ylabel('Liczba')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Tylko węzły
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(range(n), nodes, 'go-', markersize=4, label='Węzły')
    ax3.set_title('Liczba węzłów (endpoints + bifurcations)')
    ax3.set_xlabel('Klatka')
    ax3.set_ylabel('Węzły')
    ax3.grid(True, alpha=0.3)
    
    # 4. Fill vs Complexity correlation
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.scatter(selector.fill_scores * 100, selector.complexity_scores, c=range(n), cmap='viridis', s=30)
    ax4.set_xlabel('Fill [%]')
    ax4.set_ylabel('Complexity')
    ax4.set_title('Korelacja: Fill vs Complexity')
    ax4.grid(True, alpha=0.3)
    
    # 5-8. Grafy dla wybranych klatek
    sample_frames = [0, 10, 20, 30]
    sample_frames = [f for f in sample_frames if f < n]
    
    for idx, frame_idx in enumerate(sample_frames[:4]):
        ax = fig.add_subplot(gs[2, idx])
        G = selector.graphs[frame_idx]
        mask = selector.masks[frame_idx]
        
        # Pokaż maskę z węzłami grafu
        ax.imshow(mask, cmap='gray', alpha=0.5)
        
        if G.number_of_nodes() > 0:
            pos = nx.get_node_attributes(G, 'pos')
            types = nx.get_node_attributes(G, 'type')
            
            for node_id, (y, x) in pos.items():
                color = 'red' if types.get(node_id) == 'E' else 'blue'
                ax.scatter(x, y, c=color, s=50, zorder=5)
            
            for edge in G.edges():
                y1, x1 = pos[edge[0]]
                y2, x2 = pos[edge[1]]
                ax.plot([x1, x2], [y1, y2], 'yellow', linewidth=1, alpha=0.7)
        
        ax.set_title(f'Graf #{frame_idx}\nN={G.number_of_nodes()}, E={G.number_of_edges()}')
        ax.axis('off')
    
    # 9-12. Szkielety dla tych samych klatek
    for idx, frame_idx in enumerate(sample_frames[:4]):
        ax = fig.add_subplot(gs[3, idx])
        mask = selector.masks[frame_idx]
        skeleton = morphology.skeletonize(mask.astype(bool))
        ax.imshow(skeleton, cmap='gray')
        ax.set_title(f'Szkielet #{frame_idx}')
        ax.axis('off')
    
    # 13. Delta stabilności
    ax5 = fig.add_subplot(gs[4, :2])
    delta_stability = np.diff(selector.stability_scores)
    ax5.bar(range(1, n), delta_stability, color=['green' if d >= 0 else 'red' for d in delta_stability], alpha=0.7)
    ax5.set_title('Zmiana stabilności (delta)')
    ax5.set_xlabel('Klatka')
    ax5.set_ylabel('Delta')
    ax5.axhline(0, color='black', linewidth=1)
    ax5.grid(True, alpha=0.3)
    
    # 14. Histogram węzłów
    ax6 = fig.add_subplot(gs[4, 2:])
    ax6.hist(nodes, bins=15, color='blue', alpha=0.7, edgecolor='black')
    ax6.set_title('Histogram: liczba węzłów')
    ax6.set_xlabel('Węzły')
    ax6.set_ylabel('Częstość')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('DIAGNOSTYKA: Graph Matching', fontsize=14, fontweight='bold')
    plt.savefig('graph_matching_debug.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Statystyki tekstowe
    print("\n" + "="*60)
    print("STATYSTYKI GRAFÓW")
    print("="*60)
    print(f"Węzły: min={min(nodes)}, max={max(nodes)}, mean={np.mean(nodes):.1f}")
    print(f"Krawędzie: min={min(edges)}, max={max(edges)}, mean={np.mean(edges):.1f}")
    print(f"Stabilność: min={min(selector.stability_scores):.3f}, max={max(selector.stability_scores):.3f}")
    print(f"Klatki z >10 węzłami: {[i for i, n in enumerate(nodes) if n > 10]}")

# ==================== WIZUALIZACJA RUCHU PUNKTÓW KONTROLNYCH ====================

def generate_control_points_sequence(selector, start_frame=0, max_frames=None, output_dir='flow_sequence', n_points=100):
    """Generuje sekwencję klatek pokazującą ruch punktów kontrolnych przez Farneback Optical Flow (wstecz i wprzód)."""
    os.makedirs(output_dir, exist_ok=True)
    
    if max_frames is None:
        max_frames = selector.n_frames
    
    end_frame = min(start_frame + max_frames, selector.n_frames)
    
    # Znajdź klatkę z maksymalnym wypełnieniem naczynia jako źródło punktów
    fill_scores = [np.sum(m) for m in selector.masks[start_frame:end_frame]]
    best_fill_idx = np.argmax(fill_scores)
    init_frame = start_frame + best_fill_idx
    
    print(f"Klatka źródłowa punktów kontrolnych: {init_frame} (max fill)")
    
    # Dylatacja maski dla lepszego wykrywania punktów na krawędziach naczynia
    mask_dilated = cv2.dilate(selector.masks[init_frame], np.ones((5,5), np.uint8), iterations=1)
    
    # Znajdź punkty kontrolne na naczyniu używając goodFeaturesToTrack
    corners = cv2.goodFeaturesToTrack(
        selector.originals[init_frame],
        maxCorners=n_points,
        qualityLevel=0.01,
        minDistance=10,
        mask=mask_dilated,
        blockSize=7
    )
    
    if corners is None or len(corners) == 0:
        skeleton = morphology.skeletonize(selector.masks[init_frame].astype(bool))
        skel_points = np.argwhere(skeleton)
        if len(skel_points) == 0:
            print("Brak punktów kontrolnych")
            return
        indices = np.linspace(0, len(skel_points)-1, min(n_points, len(skel_points)), dtype=int)
        corners = skel_points[indices][:, ::-1].reshape(-1, 1, 2).astype(np.float32)
    
    init_points = corners.reshape(-1, 2)
    n_pts = len(init_points)
    print(f"Znaleziono {n_pts} punktów kontrolnych")
    
    # Przechowuj pozycje punktów dla każdej klatki: frame_positions[frame_idx] = array of (x, y)
    frame_positions = {init_frame: init_points.copy()}
    frame_active = {init_frame: np.ones(n_pts, dtype=bool)}
    
    # === TRACKING WSTECZ (init_frame -> start_frame) ===
    if init_frame > start_frame:
        current_points = init_points.copy()
        active = np.ones(n_pts, dtype=bool)
        
        for frame_idx in range(init_frame, start_frame, -1):
            curr_gray = selector.originals[frame_idx]
            prev_gray = selector.originals[frame_idx - 1]
            prev_mask = selector.masks[frame_idx - 1]
            
            # Optical flow wstecz
            flow = cv2.calcOpticalFlowFarneback(
                curr_gray, prev_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            new_points = current_points.copy()
            for i in range(n_pts):
                if not active[i]:
                    continue
                x, y = current_points[i]
                x_int, y_int = int(round(x)), int(round(y))
                if 0 <= y_int < flow.shape[0] and 0 <= x_int < flow.shape[1]:
                    dx, dy = flow[y_int, x_int]
                    new_x, new_y = x + dx, y + dy
                    ny, nx = int(round(new_y)), int(round(new_x))
                    if 0 <= ny < prev_mask.shape[0] and 0 <= nx < prev_mask.shape[1]:
                        y1, y2 = max(0, ny-5), min(prev_mask.shape[0], ny+6)
                        x1, x2 = max(0, nx-5), min(prev_mask.shape[1], nx+6)
                        if np.any(prev_mask[y1:y2, x1:x2]):
                            new_points[i] = [new_x, new_y]
                        else:
                            active[i] = False
                    else:
                        active[i] = False
                else:
                    active[i] = False
            
            current_points = new_points
            frame_positions[frame_idx - 1] = current_points.copy()
            frame_active[frame_idx - 1] = active.copy()
    
    # === TRACKING WPRZÓD (init_frame -> end_frame) ===
    if init_frame < end_frame - 1:
        current_points = init_points.copy()
        active = np.ones(n_pts, dtype=bool)
        
        for frame_idx in range(init_frame, end_frame - 1):
            prev_gray = selector.originals[frame_idx]
            curr_gray = selector.originals[frame_idx + 1]
            curr_mask = selector.masks[frame_idx + 1]
            
            # Optical flow wprzód
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            new_points = current_points.copy()
            for i in range(n_pts):
                if not active[i]:
                    continue
                x, y = current_points[i]
                x_int, y_int = int(round(x)), int(round(y))
                if 0 <= y_int < flow.shape[0] and 0 <= x_int < flow.shape[1]:
                    dx, dy = flow[y_int, x_int]
                    new_x, new_y = x + dx, y + dy
                    ny, nx = int(round(new_y)), int(round(new_x))
                    if 0 <= ny < curr_mask.shape[0] and 0 <= nx < curr_mask.shape[1]:
                        y1, y2 = max(0, ny-5), min(curr_mask.shape[0], ny+6)
                        x1, x2 = max(0, nx-5), min(curr_mask.shape[1], nx+6)
                        if np.any(curr_mask[y1:y2, x1:x2]):
                            new_points[i] = [new_x, new_y]
                        else:
                            active[i] = False
                    else:
                        active[i] = False
                else:
                    active[i] = False
            
            current_points = new_points
            frame_positions[frame_idx + 1] = current_points.copy()
            frame_active[frame_idx + 1] = active.copy()
    
    # === GENEROWANIE WIZUALIZACJI ===
    for frame_idx in range(start_frame, end_frame):
        img = cv2.cvtColor(selector.originals[frame_idx], cv2.COLOR_GRAY2BGR)
        mask_overlay = selector.masks[frame_idx].astype(np.uint8) * 80
        img[:, :, 1] = np.maximum(img[:, :, 1], mask_overlay)
        
        if frame_idx in frame_positions:
            points = frame_positions[frame_idx]
            active = frame_active[frame_idx]
            
            # Zbierz trajektorię (kilka ostatnich klatek)
            traj_len = 12
            traj_start = max(start_frame, frame_idx - traj_len)
            
            for i in range(n_pts):
                # Zbierz historię punktu
                history = []
                for t in range(traj_start, frame_idx + 1):
                    if t in frame_positions and t in frame_active:
                        if frame_active[t][i] or t <= init_frame:
                            history.append(frame_positions[t][i])
                
                if len(history) > 1:
                    traj = np.array(history, dtype=np.int32)
                    for j in range(len(traj) - 1):
                        alpha = (j + 1) / len(traj)
                        color = (0, int(255 * alpha), int(255 * (1-alpha)))
                        cv2.line(img, tuple(traj[j]), tuple(traj[j+1]), color, 1)
                
                # Rysuj aktualny punkt
                if active[i]:
                    x, y = points[i]
                    cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        output_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
        cv2.imwrite(output_path, img)
    
    print(f"Wygenerowano {end_frame - start_frame} klatek w {output_dir}/")

# ==================== EKSPERYMENT: DENSE FLOW - ANALIZA FAZ SERCA ====================

def analyze_dense_flow_phases(selector, grid_step=20, output_dir='dense_flow_analysis'):
    """Eksperymentalna analiza dense optical flow do wykrywania faz serca (skurcz/rozkurcz)."""
    os.makedirs(output_dir, exist_ok=True)
    
    h, w = selector.originals[0].shape
    n_frames = selector.n_frames
    
    # Metryki globalne dla każdej klatki
    mean_magnitude = []      # średnia wielkość ruchu
    mean_divergence = []     # divergence: >0 = ekspansja, <0 = kontrakcja
    div_vessel = []          # divergence tylko w obszarze naczyń
    radial_flow = []         # ruch radialny od/do środka (cały obraz)
    radial_vessel = []       # ruch radialny tylko w obszarze naczyń
    outward_ratio = []       # proporcja ruchu od centrum (0-1)
    
    # CLAHE dla wyrównania histogramu (redukuje wrażliwość na zmiany jasności)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Znajdź centrum masy naczyń (lepsze niż środek obrazu)
    all_vessel_points = []
    for mask in selector.masks:
        pts = np.argwhere(mask > 0)
        if len(pts) > 0:
            all_vessel_points.extend(pts)
    
    if all_vessel_points:
        all_vessel_points = np.array(all_vessel_points)
        cy, cx = np.mean(all_vessel_points, axis=0)
    else:
        cy, cx = h // 2, w // 2
    
    for frame_idx in range(n_frames - 1):
        prev_gray = selector.originals[frame_idx]
        curr_gray = selector.originals[frame_idx + 1]
        mask = selector.masks[frame_idx]
        
        # Preprocessing: CLAHE + lekki blur dla redukcji szumu
        prev_eq = clahe.apply(prev_gray)
        curr_eq = clahe.apply(curr_gray)
        prev_eq = cv2.GaussianBlur(prev_eq, (5, 5), 0)
        curr_eq = cv2.GaussianBlur(curr_eq, (5, 5), 0)
        
        # Dense Farneback Optical Flow na wyrównanych obrazach
        flow = cv2.calcOpticalFlowFarneback(
            prev_eq, curr_eq, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Magnitude
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_magnitude.append(np.mean(mag))
        
        # Divergence (przybliżona przez różnice pochodnych)
        du_dx = np.gradient(flow[..., 0], axis=1)
        dv_dy = np.gradient(flow[..., 1], axis=0)
        div = du_dx + dv_dy
        mean_divergence.append(np.mean(div))
        
        # Divergence tylko w obszarze naczyń
        if np.sum(mask) > 100:
            div_vessel.append(np.mean(div[mask > 0]))
        else:
            div_vessel.append(0)
        
        # Ruch radialny - cały obraz
        radial_sum = 0
        count = 0
        for y in range(0, h, grid_step):
            for x in range(0, w, grid_step):
                dx, dy = flow[y, x]
                rx, ry = x - cx, y - cy
                r_len = np.sqrt(rx**2 + ry**2)
                if r_len > 10:
                    radial = (dx * rx + dy * ry) / r_len
                    radial_sum += radial
                    count += 1
        radial_flow.append(radial_sum / max(count, 1))
        
        # Ruch radialny - tylko naczynia + proporcja od/do centrum
        radial_v_sum = 0
        count_v = 0
        outward_cnt = 0
        inward_cnt = 0
        vessel_pts = np.argwhere(mask > 0)
        for pt in vessel_pts[::5]:  # próbkuj co 5 punkt
            y, x = pt
            dx, dy = flow[y, x]
            rx, ry = x - cx, y - cy
            r_len = np.sqrt(rx**2 + ry**2)
            if r_len > 10:
                radial = (dx * rx + dy * ry) / r_len
                radial_v_sum += radial
                count_v += 1
                if radial > 0.3:
                    outward_cnt += 1
                elif radial < -0.3:
                    inward_cnt += 1
        radial_vessel.append(radial_v_sum / max(count_v, 1))
        
        # Proporcja od centrum (0-1)
        total_dir = outward_cnt + inward_cnt
        if total_dir > 50:
            outward_ratio.append(outward_cnt / total_dir)
        else:
            outward_ratio.append(0.5)  # brak danych = neutralne
    
    # Ostatnia klatka
    mean_magnitude.append(0)
    mean_divergence.append(0)
    div_vessel.append(0)
    radial_flow.append(0)
    radial_vessel.append(0)
    outward_ratio.append(0.5)
    
    mean_magnitude = np.array(mean_magnitude)
    mean_divergence = np.array(mean_divergence)
    div_vessel = np.array(div_vessel)
    radial_flow = np.array(radial_flow)
    radial_vessel = np.array(radial_vessel)
    outward_ratio = np.array(outward_ratio)
    
    # Wygładź sygnały
    from scipy.ndimage import gaussian_filter1d
    radial_smooth = gaussian_filter1d(radial_vessel, sigma=2)  # używaj radial z naczyń
    div_smooth = gaussian_filter1d(div_vessel, sigma=2)
    
    # Próg dla klasyfikacji faz (dead zone)
    # Oblicz adaptacyjny próg na podstawie amplitudy sygnału
    signal_amplitude = np.max(np.abs(radial_smooth))
    threshold = signal_amplitude * 0.15  # 15% amplitudy jako dead zone
    
    # Znajdź punkty zwrotne (przejścia przez próg)
    phase_changes = []
    for i in range(1, len(radial_smooth)):
        if radial_smooth[i-1] > threshold and radial_smooth[i] <= threshold:
            phase_changes.append(i)  # przejście do skurczu
        elif radial_smooth[i-1] < -threshold and radial_smooth[i] >= -threshold:
            phase_changes.append(i)  # przejście do rozkurczu
    
    # Próg dla divergence (używany do klasyfikacji faz)
    div_threshold = np.max(np.abs(div_smooth)) * 0.15
    
    # Znajdź zmiany faz na podstawie divergence
    phase_changes_div = []
    for i in range(1, len(div_smooth)):
        if div_smooth[i-1] > div_threshold and div_smooth[i] <= div_threshold:
            phase_changes_div.append(i)
        elif div_smooth[i-1] < -div_threshold and div_smooth[i] >= -div_threshold:
            phase_changes_div.append(i)
    
    # Lekkie wygładzenie (sigma=1) żeby nie zjeść krótkich faz
    outward_smooth = gaussian_filter1d(outward_ratio, sigma=1)
    
    # === WIZUALIZACJA 1: Wykresy faz ===
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    x = range(n_frames)
    
    axes[0].plot(x, mean_magnitude, 'b-', linewidth=1.5)
    axes[0].set_ylabel('Magnitude')
    axes[0].set_title('Średnia wielkość ruchu (optical flow z CLAHE)')
    axes[0].grid(True, alpha=0.3)
    
    # Proporcja od centrum - główny wskaźnik faz
    axes[1].plot(x, outward_ratio, 'purple', alpha=0.5, label='raw')
    axes[1].plot(x, outward_smooth, 'purple', linewidth=2, label='smooth')
    axes[1].axhline(0.5, color='black', linestyle='--', linewidth=1, label='próg 50%')
    axes[1].fill_between(x, outward_smooth, 0.5, where=(outward_smooth > 0.5), color='blue', alpha=0.2)
    axes[1].fill_between(x, outward_smooth, 0.5, where=(outward_smooth <= 0.5), color='red', alpha=0.2)
    axes[1].set_ylabel('Proporcja od centrum')
    axes[1].set_title('% punktów naczyń idących OD centrum | >50%=ROZKURCZ, ≤50%=SKURCZ')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(x, radial_vessel, 'g-', alpha=0.5, label='raw (naczynia)')
    axes[2].plot(x, radial_smooth, 'g-', linewidth=2, label='smooth')
    axes[2].axhline(0, color='black', linestyle='--', linewidth=0.5)
    axes[2].set_ylabel('Radial flow')
    axes[2].set_title('Ruch radialny naczyń (pomocniczy)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Faza serca na podstawie SUROWEGO sygnału (nie wygładzonego)
    phase = np.where(np.array(outward_ratio) > 0.5, 1, -1)
    
    # Znajdź zmiany faz
    phase_changes = []
    for i in range(1, len(phase)):
        if phase[i] != phase[i-1]:
            phase_changes.append(i)
    
    # Policz rozkurcze (ciągłe segmenty z phase=1)
    n_diastole = 0
    in_diastole = False
    for p in phase:
        if p == 1 and not in_diastole:
            n_diastole += 1
            in_diastole = True
        elif p == -1:
            in_diastole = False
    
    axes[3].fill_between(x, phase, 0, where=(phase > 0), color='blue', alpha=0.3, label='Rozkurcz')
    axes[3].fill_between(x, phase, 0, where=(phase < 0), color='red', alpha=0.3, label='Skurcz')
    for pc in phase_changes:
        axes[3].axvline(pc, color='purple', linestyle='--', alpha=0.7)
    axes[3].set_ylabel('Faza')
    axes[3].set_xlabel('Klatka')
    axes[3].set_title(f'Fazy serca | Rozkurczów: {n_diastole}, Zmian faz: {len(phase_changes)}')
    axes[3].legend()
    axes[3].set_ylim(-1.5, 1.5)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cardiac_phases.png'), dpi=150)
    plt.show()
    
    print(f"Centrum masy naczyń: ({cx:.0f}, {cy:.0f})")
    
    # === WIZUALIZACJA 2: Sekwencja z wektorami flow ===
    for frame_idx in range(n_frames - 1):
        prev_gray = selector.originals[frame_idx]
        curr_gray = selector.originals[frame_idx + 1]
        mask = selector.masks[frame_idx]
        
        # Ten sam preprocessing co w analizie
        prev_eq = clahe.apply(prev_gray)
        curr_eq = clahe.apply(curr_gray)
        prev_eq = cv2.GaussianBlur(prev_eq, (5, 5), 0)
        curr_eq = cv2.GaussianBlur(curr_eq, (5, 5), 0)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_eq, curr_eq, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Rysuj na obrazie
        img = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR)
        
        # Overlay maski naczynia
        mask_overlay = selector.masks[frame_idx + 1].astype(np.uint8) * 60
        img[:, :, 1] = np.maximum(img[:, :, 1], mask_overlay)
        
        # Policz proporcję od/do środka w obszarze naczyń
        outward_count = 0
        inward_count = 0
        vessel_pts = np.argwhere(mask > 0)
        for pt in vessel_pts[::5]:
            y, x = pt
            dx, dy = flow[y, x]
            rx, ry = x - cx, y - cy
            r_len = np.sqrt(rx**2 + ry**2)
            if r_len > 10:
                radial = (dx * rx + dy * ry) / r_len
                if radial > 0.3:
                    outward_count += 1
                elif radial < -0.3:
                    inward_count += 1
        
        # Rysuj wektory flow na regularnej siatce
        for y in range(grid_step//2, h, grid_step):
            for x in range(grid_step//2, w, grid_step):
                dx, dy = flow[y, x]
                flow_mag = np.sqrt(dx**2 + dy**2)
                if flow_mag > 0.5:
                    scale = 3
                    end_x = int(x + dx * scale)
                    end_y = int(y + dy * scale)
                    rx, ry = x - cx, y - cy
                    r_len = np.sqrt(rx**2 + ry**2)
                    if r_len > 10:
                        radial = (dx * rx + dy * ry) / r_len
                        if radial > 0:
                            color = (255, 100, 100)  # niebieski - od środka
                        else:
                            color = (100, 100, 255)  # czerwony - do środka
                    else:
                        color = (200, 200, 200)
                    cv2.arrowedLine(img, (x, y), (end_x, end_y), color, 1, tipLength=0.3)
        
        # Rysuj centrum masy naczyń (zielone kółko z etykietą)
        cv2.circle(img, (int(cx), int(cy)), 8, (0, 255, 0), 2)
        cv2.circle(img, (int(cx), int(cy)), 2, (0, 255, 0), -1)
        cv2.putText(img, "C", (int(cx)+10, int(cy)+5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Klasyfikacja na podstawie PROPORCJI od/do centrum w naczyniach
        total_directional = outward_count + inward_count
        if total_directional > 50:
            ratio = outward_count / total_directional
            if ratio > 0.5:  # >50% od centrum = ROZKURCZ
                phase_str = "ROZKURCZ"
                phase_color = (255, 200, 100)
            else:  # <=50% = SKURCZ
                phase_str = "SKURCZ"
                phase_color = (100, 100, 255)
        else:
            phase_str = "?"
            phase_color = (128, 128, 128)
            ratio = 0.5
        
        cv2.putText(img, f"Frame {frame_idx}: {phase_str}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 2)
        cv2.putText(img, f"Od centrum: {ratio*100:.0f}%", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"Naczynia: {outward_count} od / {inward_count} do", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        output_path = os.path.join(output_dir, f'flow_{frame_idx:04d}.png')
        cv2.imwrite(output_path, img)
    
    print(f"\nAnaliza faz serca:")
    print(f"  Wykryte zmiany faz w klatkach: {phase_changes}")
    print(f"  Liczba cykli (przybliżona): {len(phase_changes) // 2}")
    print(f"Wygenerowano wizualizacje w {output_dir}/")
    
    return {
        'magnitude': mean_magnitude,
        'divergence': mean_divergence,
        'radial_flow': radial_flow,
        'radial_smooth': radial_smooth,
        'phase_changes': phase_changes
    }

# ==================== WIZUALIZACJA ====================

def visualize_results(masks, result, method_name, filenames=None):
    best_idx = result['best_frame']
    metrics = result['metrics']
    n_frames = len(masks)
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    
    ax1 = fig.add_subplot(gs[0, :])
    x = range(n_frames)
    
    colors = ['blue', 'orange', 'green', 'purple', 'brown']
    for i, (name, values) in enumerate(metrics.items()):
        ax1.plot(x, values, label=name.capitalize(), color=colors[i % len(colors)], alpha=0.7, linewidth=1.5)
    
    ax1.plot(x, result['scores'], label='SCORE', color='black', linewidth=2.5, linestyle='--')
    ax1.axvline(best_idx, color='red', linestyle='-', linewidth=2, label=f'Wybrana: {best_idx}')
    ax1.set_title(f"{method_name} - Analiza sekwencji", fontsize=14)
    ax1.set_xlabel("Numer klatki")
    ax1.set_ylabel("Znormalizowana wartość")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, n_frames-1)
    
    # Wybrana klatka i sąsiednie
    prev_idx = max(0, best_idx - 5)
    next_idx = min(n_frames - 1, best_idx + 5)
    
    for ax_pos, frame_idx, is_winner in [(gs[1,0], prev_idx, False),
                                          (gs[1,1], best_idx, True),
                                          (gs[1,2], next_idx, False)]:
        ax = fig.add_subplot(ax_pos)
        ax.imshow(masks[frame_idx], cmap='gray')
        
        if is_winner:
            title = f"WINNER: Frame {frame_idx}\nScore: {result['scores'][frame_idx]:.3f}"
            ax.set_title(title, fontsize=12, fontweight='bold', color='green')
        else:
            ax.set_title(f"Frame {frame_idx}", fontsize=11)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{method_name.replace(" ", "_").lower()}_result.png', dpi=150)
    plt.show()
    
    if filenames:
        print(f"Wybrana klatka: {best_idx} -> {filenames[best_idx]}")

# ==================== MAIN ====================

if __name__ == "__main__":
    base_dir = '/Users/rafalszulinski/Desktop/developing/IVES/coronary/keyframe/keyframe-selector/1-100/10_I0572687.VIM.DCM'
    
    masks, originals, filenames = load_data_from_directory(base_dir)
    
    if not masks:
        print("Brak danych")
        exit()
    
    print(f"\n{'='*60}")
    print("METODA 1: Masked Optical Flow")
    print('='*60)
    
    of_selector = OpticalFlowKeyframeSelector(masks, originals)
    of_result = of_selector.select_best_frame(w_fill=0.4, w_thick=0.4, w_stability=0.0, w_phase=0.2)
    print(f"Najlepsza klatka: {of_result['best_frame']} ({filenames[of_result['best_frame']]})")
    print(f"Score: {of_result['scores'][of_result['best_frame']]:.4f}")
    visualize_results(masks, of_result, "Optical Flow", filenames)
    
    # Diagnostyka Optical Flow
    print("\n--- Diagnostyka Optical Flow ---")
    visualize_optical_flow_debug(of_selector, of_result)
    
    # print(f"\n{'='*60}")
    # print("METODA 2: Graph Matching")
    # print('='*60)
    
    # gm_selector = GraphMatchingKeyframeSelector(masks)
    # gm_result = gm_selector.select_best_frame(w_stability=0.3, w_complexity=0.3, w_fill=0.4)
    # print(f"Najlepsza klatka: {gm_result['best_frame']} ({filenames[gm_result['best_frame']]})")
    # print(f"Score: {gm_result['scores'][gm_result['best_frame']]:.4f}")
    # visualize_results(masks, gm_result, "Graph Matching", filenames)
    
    # # Diagnostyka Graph Matching
    # print("\n--- Diagnostyka Graph Matching ---")
    # visualize_graph_debug(gm_selector, gm_result)
    
    # print(f"\n{'='*60}")
    # print("PORÓWNANIE METOD")
    # print('='*60)
    # print(f"Optical Flow: Frame {of_result['best_frame']}")
    # print(f"Graph Matching: Frame {gm_result['best_frame']}")
    
    # if of_result['best_frame'] == gm_result['best_frame']:
    #     print("Obie metody wskazują tę samą klatkę!")
    
    print(f"\n{'='*60}")
    print("GENEROWANIE SEKWENCJI Z RUCHEM PUNKTÓW KONTROLNYCH")
    print('='*60)
    generate_control_points_sequence(of_selector, start_frame=0, max_frames=None, output_dir='flow_sequence')
    
    print(f"\n{'='*60}")
    print("EKSPERYMENT: DENSE FLOW - ANALIZA FAZ SERCA")
    print('='*60)
    phase_result = analyze_dense_flow_phases(of_selector, grid_step=15, output_dir='dense_flow_analysis')
