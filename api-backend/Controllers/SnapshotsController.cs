using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using api_backend.Data;
using api_backend.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace api_backend.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class SnapshotsController : ControllerBase
    {
        private readonly AppDbContext _context;

        public SnapshotsController(AppDbContext context)
        {
            _context = context;
        }

        // GET: api/snapshots
        [HttpGet]
        public async Task<ActionResult<IEnumerable<object>>> GetSnapshots(
            [FromQuery] DateTime? startDate = null, 
            [FromQuery] DateTime? endDate = null)
        {
            var query = _context.CurrencySnapshots.AsQueryable();

            if (startDate.HasValue)
                query = query.Where(s => s.SnapshotTime >= startDate.Value);

            if (endDate.HasValue)
                query = query.Where(s => s.SnapshotTime <= endDate.Value);

            var snapshots = await query
                .OrderByDescending(s => s.SnapshotTime)
                .Select(s => new
                {
                    s.Id,
                    s.SnapshotTime,
                    SnapshotData = s.SnapshotData
                })
                .ToListAsync();

            return snapshots;
        }

        // GET: api/snapshots/5
        [HttpGet("{id}")]
        public async Task<ActionResult<object>> GetSnapshot(int id)
        {
            var snapshot = await _context.CurrencySnapshots
                .Where(s => s.Id == id)
                .Select(s => new
                {
                    s.Id,
                    s.SnapshotTime,
                    SnapshotData = s.SnapshotData
                })
                .FirstOrDefaultAsync();

            if (snapshot == null)
            {
                return NotFound();
            }

            return snapshot;
        }
    }
}