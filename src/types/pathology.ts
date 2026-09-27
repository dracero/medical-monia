export interface HistologySample {
  id: string;
  title: string;
  organ: string;
  stain: string;
  magnification: string;
  resolution: string;
  file: string;
  description: string;
}

export interface PromptPoint {
  x: number;
  y: number;
  label: 1 | 0; // 1: foreground (green), 0: background (red)
}

export interface ZeroShotAnnotation {
  id: string;
  name: string;
  polygon: [number, number][];
  color: string;
  areaUm2: number;
  confidence: number;
  timestamp: string;
  visible: boolean;
}

export interface MacroUnit {
  id: string;
  name: string;
  center: [number, number];
  rx: number;
  ry: number;
  area_um2: number;
  circularity: number;
}

export interface MacroLumen {
  id: string;
  name: string;
  center: [number, number];
  rx: number;
  ry: number;
  area_um2: number;
}

export interface MicroLayer {
  name: string;
  count: number;
  color: string;
  percentage: number;
}

export interface MicroStructureData {
  total_nuclei_detected: number;
  mean_nuclear_diameter_um: number;
  cellular_density_mm2: number;
  layers: MicroLayer[];
}

export interface UniversalAnalysisResult {
  status: string;
  sample_id: string;
  engine: string;
  macrostructure: {
    tissue_units: MacroUnit[];
    lumens: MacroLumen[];
    total_units_count: number;
    parenchyma_stroma_ratio: Record<string, string>;
  };
  microstructure: MicroStructureData;
  physical_calibration_um_per_px: number;
}

export type ViewScale = 'all' | 'macro' | 'micro';
export type ViewerTool = 'prompt-pos' | 'prompt-neg' | 'box' | 'pan';

export interface TextPromptPolygon {
  id: string;
  center: [number, number];
  area_px: number;
  polygon: [number, number][];
}

export interface TextPromptResult {
  success?: boolean;
  status?: string;
  prompt?: string;
  translated_prompt?: string;
  prompt_original?: string;
  prompt_translated?: string;
  prompt_clip?: string;
  sensitivity?: number;
  tissue_masking?: boolean;
  normalize_stain?: boolean;
  overlap?: number;
  device?: string;
  inference_engine?: string;
  session_id?: string;
  image_size?: { width: number; height: number };
  image_dimensions?: { width: number; height: number };
  heatmap_png_base64?: string;
  heatmap_png?: string;
  mask_png_base64?: string;
  mask_png?: string;
  polygons?: any[];
  metrics?: {
    segmented_pixels?: number;
    total_segmented_px?: number;
    tissue_pixels?: number;
    coverage_pct?: number;
    percentage_of_image?: number;
    area_um2?: number;
    area_mm2?: number;
    region_count?: number;
    num_regions?: number;
    mean_confidence?: number;
    elapsed_seconds?: number;
  };
  calibration_um_per_px?: number;
}
