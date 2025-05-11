import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ClusterChange {
  id: number;
  currencyId: number;
  symbol: string;
  fromClusterId: number;
  toClusterId: number;
  changeTimestamp: string;
}

@Injectable({
  providedIn: 'root'
})
export class GetClusterChangesService {
  private apiUrl = 'http://localhost:5101/api/ClusterChanges';

  constructor(private http: HttpClient) { }

  getClusterChanges(symbol: string): Observable<ClusterChange[]> {
    return this.http.get<ClusterChange[]>(`${this.apiUrl}/${symbol}`);
  }
}
