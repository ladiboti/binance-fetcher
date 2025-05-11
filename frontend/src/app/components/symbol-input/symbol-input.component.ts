import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { RunClusteringPipelineService } from '../../services/run-clustering-pipeline.service';

@Component({
  selector: 'app-symbol-input',
  standalone: false, // Változtasd meg erre
  templateUrl: './symbol-input.component.html',
  styleUrls: ['./symbol-input.component.css']
})
export class SymbolInputComponent {
  symbols: string[] = [];
  newSymbol: string = '';
  errorMessage: string = '';
  isLoading: boolean = false;

  constructor(
    private clusteringService: RunClusteringPipelineService,
    private router: Router
  ) { }

  onInputChange(value: string) {
    this.newSymbol = value;
    if (this.errorMessage && value.trim()) {
      this.errorMessage = '';
    }
  }

  addSymbol() {
    console.log('Adding symbol:', this.newSymbol);
    
    const symbol = this.newSymbol.trim().toUpperCase();
    
    if (!symbol) {
      this.errorMessage = 'Symbol cannot be empty';
      return;
    }
    
    if (this.symbols.includes(symbol)) {
      this.errorMessage = 'Symbol already exists';
      return;
    }
    
    this.symbols.push(symbol);
    this.newSymbol = '';
    this.errorMessage = '';
  }
  
  removeSymbol(symbol: string) {
    this.symbols = this.symbols.filter(s => s !== symbol);
  }

  fetchSymbols() {
    if (this.symbols.length === 0) {
      this.errorMessage = 'Please add at least one symbol';
      return;
    }
    
    this.isLoading = true;
    
    this.clusteringService.fetchSymbols(this.symbols)
      .subscribe({
        next: (response) => {
          console.log('Pipeline completed:', response);
          this.isLoading = false;
          this.router.navigate(['/analysis'])
        },
        error: (error) => {
          console.error('Error running pipeline:', error);
          this.errorMessage = 'Failed to run clustering pipeline';
          this.isLoading = false;
        }
      });
  }
}
