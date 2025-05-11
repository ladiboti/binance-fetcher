import { Component, OnInit } from '@angular/core';
import { GetCurrenciesService} from '../../services/get-currencies.service';
import { GetClusterChangesService, ClusterChange } from '../../services/get-cluster-changes.service';

interface Currency {
  id: number;
  symbol: string;
  name: string;
  clusterId: number;
  totalClusterChanges: number;
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
  clusterChanges: ClusterChange[] = [];
  selectedSymbol: string | null = null;
  errorMessage: string = '';

  constructor(
    private currencyService: GetCurrenciesService,
    private clusterChangeService: GetClusterChangesService
  ) { }

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

  loadClusterChanges(symbol: string): void {
    this.selectedSymbol = symbol;
    this.clusterChangeService.getClusterChanges(symbol).subscribe({
      next: (data) => this.clusterChanges = data,
      error: (error) => {
        console.error('Error fetching cluster changes:', error);
        this.errorMessage = 'Failed to fetch cluster changes';
      }
    });
  }

  closeClusterChanges(): void {
    this.selectedSymbol = null;
    this.clusterChanges = [];
  }
}
