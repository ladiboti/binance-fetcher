import { TestBed } from '@angular/core/testing';

import { GetClusterChangesService } from './get-cluster-changes.service';

describe('GetClusterChangesService', () => {
  let service: GetClusterChangesService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(GetClusterChangesService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
