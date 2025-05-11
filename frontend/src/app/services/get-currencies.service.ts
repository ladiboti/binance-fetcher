import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

interface Currency {
  id: number;
  symbol: string;
  name: string;
  clusterId: number;
  totalClusterChanges: number;
  createdAt: string;
}

@Injectable({
  providedIn: 'root'
})
export class GetCurrenciesService {
  private apiUrl = 'http://localhost:5101/api/Currencies';

  constructor(private http: HttpClient) { }

  getCurrencies(): Observable<Currency[]> {
    return this.http.get<Currency[]>(this.apiUrl);
  }
}
