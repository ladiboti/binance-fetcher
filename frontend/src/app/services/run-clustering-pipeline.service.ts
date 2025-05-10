import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { catchError } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class RunClusteringPipelineService {
  private apiUrl = 'http://localhost:5000/api';

  constructor(private http: HttpClient) { }

  fetchSymbols(symbols: string[], interval: string = '1h', days: number = 30, limit: number = 1000): Observable<any> {
    const requestData = {
      trading_pairs: symbols,
      interval,
      days,
      limit
    };

    return this.http.post<any>(`${this.apiUrl}/fetch`, requestData);
  }
}
