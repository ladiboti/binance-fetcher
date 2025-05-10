import { TestBed } from '@angular/core/testing';

import { RunClusteringPipelineService } from './run-clustering-pipeline.service';

describe('RunClusteringPipelineService', () => {
  let service: RunClusteringPipelineService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(RunClusteringPipelineService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
