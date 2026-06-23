/**
 * API client for communicating with the FastAPI backend.
 * All requests are proxied through Vite dev server to avoid CORS.
 */

const BASE_URL = '/api';

interface FetchOptions extends RequestInit {
  params?: Record<string, string | number | undefined | null>;
}

async function request<T>(endpoint: string, options: FetchOptions = {}): Promise<T> {
  const { params, ...fetchOptions } = options;

  let url = `${BASE_URL}${endpoint}`;

  // Append query params
  if (params) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') {
        searchParams.append(key, String(value));
      }
    });
    const qs = searchParams.toString();
    if (qs) url += `?${qs}`;
  }

  const response = await fetch(url, {
    ...fetchOptions,
    headers: {
      ...(fetchOptions.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...fetchOptions.headers,
    },
  });

  if (!response.ok) {
    const errorBody = await response.text();
    let detail = `HTTP ${response.status}`;
    try {
      const parsed = JSON.parse(errorBody);
      detail = parsed.detail || detail;
    } catch {
      // ignore parse error
    }
    throw new Error(detail);
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

// ─── Types ──────────────────────────────────────────────────────────────────

export interface PersonFace {
  id: number;
  person_id: number;
  image_url: string;
  created_at: string;
}

export interface Person {
  id: number;
  name: string;
  department: string;
  notes: string;
  face_count: number;
  faces: PersonFace[];
  created_at: string;
  updated_at: string;
}

export interface PersonList {
  items: Person[];
  total: number;
  page: number;
  limit: number;
}

export interface Camera {
  id: number;
  name: string;
  url: string;
  location: string;
  is_active: boolean;
  fps_process: number;
  detect_mode: 'face' | 'plate' | 'both';
  created_at: string;
}

export interface CameraList {
  items: Camera[];
  total: number;
}

export interface Detection {
  id: number;
  person_id: number | null;
  person_name: string;
  camera_id: number;
  camera_name: string;
  confidence: number;
  snapshot_url: string | null;
  face_crop_url: string | null;
  restored_face_url: string | null;
  detected_at: string;
}

export interface DetectionList {
  items: Detection[];
  total: number;
  page: number;
  limit: number;
}

export interface DetectionStats {
  total_detections: number;
  unique_persons: number;
  by_camera: Record<string, number>;
  by_hour: Record<string, number>;
}

export interface StatsOverview {
  total_cameras: number;
  active_cameras: number;
  total_persons: number;
  total_detections_today: number;
}

export interface WorkerCameraInfo {
  id: number;
  name: string;
}

export interface WorkerInfo {
  id: string;
  connected_at: string;
  uptime: string;
  cameras: WorkerCameraInfo[];
  avg_process_ms: number;
  is_paused: boolean;
}

export interface WorkerList {
  workers: WorkerInfo[];
  total: number;
}

export interface DetectionEvent {
  type: string;
  person_id: number | null;
  person_name: string;
  camera_id: number;
  camera_name: string;
  confidence: number;
  snapshot_url: string | null;
  face_crop_url: string | null;
  restored_face_url: string | null;
  timestamp: string;
}

// ─── API Functions ──────────────────────────────────────────────────────────

export const api = {
  // Health
  health: () => request<{ status: string }>('/health'),

  // Persons
  listPersons: (params?: { search?: string; page?: number; limit?: number }) =>
    request<PersonList>('/persons', { params: params as Record<string, string | number> }),

  createPerson: (data: { name: string; department?: string; notes?: string }) =>
    request<Person>('/persons', { method: 'POST', body: JSON.stringify(data) }),

  getPerson: (id: number) =>
    request<Person>(`/persons/${id}`),

  updatePerson: (id: number, data: { name?: string; department?: string; notes?: string }) =>
    request<Person>(`/persons/${id}`, { method: 'PUT', body: JSON.stringify(data) }),

  deletePerson: (id: number) =>
    request<void>(`/persons/${id}`, { method: 'DELETE' }),

  uploadFaces: (personId: number, files: File[]) => {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));
    return request<PersonFace[]>(`/persons/${personId}/faces`, {
      method: 'POST',
      body: formData,
    });
  },

  deleteFace: (personId: number, faceId: number) =>
    request<void>(`/persons/${personId}/faces/${faceId}`, { method: 'DELETE' }),

  // Cameras
  listCameras: () =>
    request<CameraList>('/cameras'),

  createCamera: (data: { name: string; url: string; location?: string; fps_process?: number; detect_mode?: string }) =>
    request<Camera>('/cameras', { method: 'POST', body: JSON.stringify(data) }),

  updateCamera: (id: number, data: { name?: string; url?: string; location?: string; fps_process?: number; detect_mode?: string }) =>
    request<Camera>(`/cameras/${id}`, { method: 'PUT', body: JSON.stringify(data) }),

  deleteCamera: (id: number) =>
    request<void>(`/cameras/${id}`, { method: 'DELETE' }),

  startCamera: (id: number) =>
    request<{ status: string; message: string }>(`/cameras/${id}/start`, { method: 'POST' }),

  stopCamera: (id: number) =>
    request<{ status: string; message: string }>(`/cameras/${id}/stop`, { method: 'POST' }),

  getCameraSnapshotUrl: (id: number) =>
    `${BASE_URL}/cameras/${id}/snapshot?t=${Date.now()}`,

  // Detections
  listDetections: (params?: {
    person_id?: number;
    camera_id?: number;
    date_from?: string;
    date_to?: string;
    page?: number;
    limit?: number;
  }) =>
    request<DetectionList>('/detections', { params: params as Record<string, string | number> }),

  getDetectionStats: (params?: { date_from?: string; date_to?: string }) =>
    request<DetectionStats>('/detections/stats', { params: params as Record<string, string | number> }),

  getOverview: () =>
    request<StatsOverview>('/detections/overview'),

  // License Plate Detections
  listPlateDetections: (params?: {
    camera_id?: number;
    date_from?: string;
    date_to?: string;
    page?: number;
    limit?: number;
  }) =>
    request<PlateDetectionList>('/plate-detections', { params: params as Record<string, string | number> }),

  getPlateDetectionStats: () =>
    request<PlateDetectionStats>('/plate-detections/stats'),

  clearPlateDetections: () =>
    request<{ deleted: number }>('/plate-detections', { method: 'DELETE' }),

  // Workers
  listWorkers: () =>
    request<WorkerList>('/workers'),

  toggleWorkerPause: (id: string) =>
    request<{ id: string; is_paused: boolean }>(`/workers/${id}/toggle-pause`, { method: 'POST' }),

  // Surveillance Station
  ssTestConnection: (data: SSConnectRequest) =>
    request<{ status: string; message: string; camera_count: number }>(
      '/surveillance-station/test',
      { method: 'POST', body: JSON.stringify(data) }
    ),

  ssListCameras: (data: SSConnectRequest) =>
    request<SSCameraListResponse>(
      '/surveillance-station/cameras',
      { method: 'POST', body: JSON.stringify(data) }
    ),

  ssImportCameras: (data: SSImportRequest) =>
    request<SSImportResult>(
      '/surveillance-station/import',
      { method: 'POST', body: JSON.stringify(data) }
    ),
};

// ─── Surveillance Station Types ─────────────────────────────────────────────

export interface SSConnectRequest {
  base_url: string;
  username: string;
  password: string;
  verify_ssl: boolean;
}

export interface SSCameraInfo {
  ss_id: number;
  name: string;
  model: string;
  host: string;
  port: number;
  status: number;
  enabled: boolean;
  vendor: string;
  resolution: string;
  rtsp_url: string;
  mjpeg_url: string;
  already_imported: boolean;
}

export interface SSCameraListResponse {
  cameras: SSCameraInfo[];
  total: number;
  nas_url: string;
}

export interface SSImportRequest extends SSConnectRequest {
  camera_ids: number[];
}

export interface SSImportResult {
  imported: number;
  skipped: number;
  errors: string[];
  cameras: { id: number; name: string; url: string; location: string; ss_id: number }[];
}

// ─── License Plate Types ─────────────────────────────────────────────────────

export interface PlateDetection {
  id: number;
  camera_id: number;
  camera_name: string;
  plate_number: string;
  raw_text: string;
  confidence: number;
  plate_type: string;
  province: string;
  snapshot_url: string | null;
  detected_at: string;
}

export interface PlateDetectionList {
  items: PlateDetection[];
  total: number;
  page: number;
  limit: number;
}

export interface PlateDetectionStats {
  total_today: number;
  total_all: number;
  by_camera: { camera_name: string; count: number }[];
}

export interface PlateDetectionEvent {
  type: 'plate_detection';
  camera_id: number;
  camera_name: string;
  plate_number: string;
  raw_text: string;
  confidence: number;
  plate_type: string;
  province: string;
  snapshot_url: string | null;
  timestamp: string;
}

// ─── Training Sample Types ────────────────────────────────────────────────────

export interface CharLabel {
  class_name: string;
  cx: number;
  cy: number;
  bw: number;
  bh: number;
  confidence: number;
}

export interface TrainingSample {
  id: number;
  camera_id: number;
  camera_name: string;
  track_id: string;
  image_url: string;
  char_labels: string; // JSON string → CharLabel[]
  raw_text: string;
  corrected_text: string;
  confidence: number;
  status: 'pending' | 'approved' | 'rejected';
  detected_at: string;
  created_at: string;
}

export interface TrainingSampleList {
  items: TrainingSample[];
  total: number;
  page: number;
  limit: number;
}

export interface TrainingStats {
  by_status: { status: string; count: number }[];
  by_class: { class_name: string; count: number }[];
  total_pending: number;
}

export interface ExportPreview {
  total: number;
  status: string;
  conf_max: string;
}

export const trainingApi = {
  list: (params: {
    page?: number;
    limit?: number;
    status?: string;
    conf_max?: number;
    conf_min?: number;
    camera_id?: number;
  } = {}): Promise<TrainingSampleList> =>
    request('/training/plates', { params: params as Record<string, string | number | undefined | null> }),

  get: (id: number): Promise<TrainingSample> =>
    request(`/training/plates/${id}`),

  update: (id: number, body: { status?: string; corrected_text?: string; char_labels?: string }): Promise<TrainingSample> =>
    request(`/training/plates/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  bulkUpdate: (ids: number[], status: string): Promise<{ updated: number }> =>
    request('/training/plates/bulk', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids, status }),
    }),

  approveTrack: (trackId: string, status: string): Promise<{ updated: number; track_id: string }> =>
    request(`/training/plates/track/${encodeURIComponent(trackId)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    }),

  stats: (): Promise<TrainingStats> =>
    request('/training/plates/stats'),

  exportPreview: (status = 'approved', conf_max?: number): Promise<ExportPreview> =>
    request('/training/plates/export/preview', {
      params: { status, conf_max: conf_max ?? null },
    }),

  clear: (status?: string): Promise<{ deleted: number }> =>
    request('/training/plates', {
      method: 'DELETE',
      params: status ? { status } : {},
    }),

  exportUrl: (status = 'approved', conf_max?: number): string => {
    const p = new URLSearchParams({ status });
    if (conf_max != null) p.set('conf_max', String(conf_max));
    return `/api/training/plates/export?${p.toString()}`;
  },
};
