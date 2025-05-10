import { Component } from '@angular/core';

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
}
