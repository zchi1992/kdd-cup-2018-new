import math
import numpy as np
from multiprocessing.pool import Pool
import warnings

class STMVL:
    
    def __init__(self, alpha=4, gamma=0.85, window_size=7, temporal_threshold=5,
                 is_initialize=True, n_jobs=None):
        self.alpha = alpha
        self.gamma = gamma
        self.window_size = window_size
        self.temporal_threshold = temporal_threshold
        self.is_initialize = is_initialize
        self.n_jobs = n_jobs
        
    def fit_transform(self, data, latitude, longitude):
        print('STMVL: Preparing')
        self.row_count, self.column_count = data.shape
        self.missing_matrix = data.copy()
        self.temporary_matrix = data.copy()
        self.distances = np.zeros([self.column_count, self.column_count])
        for i in range(self.column_count):
            for j in range(self.column_count):
                self.distances[i, j] = self.geo_distance(latitude[i], longitude[i], 
                                                         latitude[j], longitude[j])
        if self.is_initialize:
            self._initialize_missing()
        with Pool(self.n_jobs) as p:
            print('STMVL: Start with', p._processes, 'processes')
            return np.stack([*p.map(self._fit_transform, range(self.column_count))], axis=1)
        
    def _initialize_missing(self):
        for i in range(self.row_count):
            for j in range(self.column_count):
                if np.isnan(self.missing_matrix.item(i, j)):
                    self.global_view_combine(i, j)
    
    def global_view_combine(self, i, j):
        IDW = self._IDW(i, j, self.missing_matrix)
        SES = self._SES(i, j, self.missing_matrix)
        if SES and IDW:
            self.temporary_matrix[i, j] = (IDW + SES) / 2
        elif SES:
            self.temporary_matrix[i, j] = SES
        elif IDW:
            self.temporary_matrix[i, j] = IDW
            
    def _fit_transform(self, j):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            X = []
            y = []
            for i in range(self.row_count):
                if not np.isnan(self.missing_matrix.item(i, j)):
                    if self._check_context_data(i, j):
                        views = self._views(i, j)
                        if all(views):
                            X.append(views)
                            y.append(self.missing_matrix.item(i, j))
            X = np.array(X)
            y = np.array(y)
            w, ls, _, _ = np.linalg.lstsq(np.c_[X, np.ones([len(X), 1])], y, rcond=-1)
            result = np.zeros(self.row_count)
            for i in range(self.row_count):
                if np.isnan(self.missing_matrix[i, j]):
                    views = self._views(i, j)
                    if all(views):
                        result[i] = np.dot(w, [*views, 1])
                    else:
                        result[i] = self.temporary_matrix[i, j]
                else:
                    result[i] = self.missing_matrix[i, j]
            print('STMVL:', 'Sensor', j, w, 'RMSE', math.sqrt(ls / len(X)))
            print(result)
            return result
                    
    def _check_context_data(self, i, j):
        data = self.missing_matrix[i, :]
        if np.isnan(data).sum() > len(data) / 2:
            return False
        data = self.missing_matrix[max(0, i - self.window_size//2):
                                   i + self.window_size//2 + 1, j]
        if np.isnan(data).sum() > len(data) / 2:
            return False
        return True
    
    def _views(self, i, j):
        return (self._UCF(i, j, self.temporary_matrix), self._ICF(i, j, self.temporary_matrix), 
                self._IDW(i, j, self.temporary_matrix), self._SES(i, j, self.temporary_matrix)) 
    
    def _UCF(self, i, j, data_matrix):
        backup = data_matrix[i, j]
        data_matrix[i, j] = np.nan
        sim_sum = 0
        val_sum = 0
        s = max(0, i - self.window_size // 2)
        t = min(self.row_count, i + self.window_size//2 + 1)
        for k in range(self.column_count):
            if j != k and not np.isnan(data_matrix.item(i, k)):
                offset = 0
                NT = 0
                for l in range(s, t):
                    temp = (data_matrix.item(l, j) - data_matrix.item(l, k))**2
                    if not np.isnan(temp):
                        offset += temp
                        NT += 1
                if NT == 0 or offset == 0:
                    continue
                sim = 1 / math.sqrt(offset / NT)
                sim_sum += sim
                val_sum += sim * data_matrix.item(i, k)
        data_matrix[i, j] = backup
        return val_sum / sim_sum if sim_sum > 0 else None
    
    def _ICF(self, i, j, data_matrix):
        backup = data_matrix[i, j]
        data_matrix[i, j] = np.nan
        sim_sum = 0
        val_sum = 0
        s = max(0, i - self.window_size // 2)
        t = min(self.row_count, i + self.window_size//2 + 1)
        for k in range(s, t):
            if i != k and not np.isnan(data_matrix.item(k, j)):
                sim_inv = math.sqrt(np.nanmean((data_matrix[i, :] - data_matrix[k, :]) ** 2))
                if np.isnan(sim_inv) or sim_inv == 0:
                    continue
                sim = 1 / sim_inv
                sim_sum += sim
                val_sum += sim * data_matrix.item(k, j)
        data_matrix[i, j] = backup
        return val_sum / sim_sum if sim_sum > 0 else None
    
    def _SES(self, i, j, data_matrix):
        sim_sum = 0
        val_sum = 0
        s = max(0, i - self.temporal_threshold)
        t = min(self.row_count, i + self.temporal_threshold + 1)
        for k in range(s, t):
            if i != k and not np.isnan(data_matrix.item(k, j)):
                sim = self.gamma * (1 - self.gamma)**(abs(i - k) - 1)
                sim_sum += sim
                val_sum += sim * data_matrix.item(k, j)
        return val_sum / sim_sum if sim_sum > 0 else None
    
    def _IDW(self, i, j, data_matrix):
        sim_sum = 0
        val_sum = 0
        for k in range(self.column_count):
            if j != k and not np.isnan(data_matrix.item(i, k)):
                sim = (1 / self.distances.item(j, k))**self.alpha
                sim_sum += sim
                val_sum += sim * data_matrix.item(i, k)
        return val_sum / sim_sum if sim_sum > 0 else None
        
    @staticmethod
    def geo_distance(lat1, lng1, lat2, lng2):
        EARTH_RADIUS = 6378137.0
        rad_lat1 = math.radians(lat1)
        rad_lat2 = math.radians(lat2)
        a = rad_lat1 - rad_lat2
        b = math.radians(lng1) - math.radians(lng2)
        s = 2 * math.asin(math.sqrt(math.sin(a/2)**2 + math.cos(rad_lat1) * 
                                    math.cos(rad_lat2) * math.sin(b/2)**2))
        if s < 0.000001:
            s = 0.000001
        return s * EARTH_RADIUS
