import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

import { SymbolInputComponent } from './components/symbol-input/symbol-input.component';
import { AnalysisPageComponent } from './components/analysis-page/analysis-page.component';

const routes: Routes = [
  { path: '', component: SymbolInputComponent, pathMatch: 'full' },
  { path: 'analysis', component: AnalysisPageComponent }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
