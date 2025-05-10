import { NgModule } from '@angular/core';
import { BrowserModule, provideClientHydration, withEventReplay } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import { provideHttpClient, withInterceptorsFromDi } from '@angular/common/http';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { SymbolInputComponent } from './components/symbol-input/symbol-input.component';

@NgModule({
  declarations: [
    AppComponent,
    SymbolInputComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule, 
    FormsModule,
  ],
  providers: [
    provideClientHydration(withEventReplay()),
    provideHttpClient(withInterceptorsFromDi()),
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
