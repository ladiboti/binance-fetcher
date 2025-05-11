import { Component, OnInit } from '@angular/core';
import { GetCurrenciesService } from '../../services/get-currencies.service';

interface Currency {
  id: number;
  symbol: string;
  name: string;
  clusterId: number;
  createdAt: string;
}

@Component({
  selector: 'app-analysis-page',
  standalone: false,
  templateUrl: './analysis-page.component.html',
  styleUrl: './analysis-page.component.css'
})
export class AnalysisPageComponent {
  currencies: Currency[] = [];
  errorMessage: string = '';

  constructor(private currencyService: GetCurrenciesService) { }

  ngOnInit(): void {
    this.fetchCurrencies();
  }

  fetchCurrencies(): void {
    this.currencyService.getCurrencies().subscribe({
      next: (data) => this.currencies = data,
      error: (error) => {
        console.error('Error fetching currencies:', error);
        this.errorMessage = 'Failed to fetch currencies';
      }
    });
  }
}
