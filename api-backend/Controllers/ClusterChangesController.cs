using api_backend.Data;
using api_backend.Models;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace api_backend.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ClusterChangesController : ControllerBase
    {
        private readonly AppDbContext _context;

        public ClusterChangesController(AppDbContext context)
        {
            _context = context;
        }

        // GET: api/ClusterChanges/{symbol}
        [HttpGet("{symbol}")]
        public async Task<ActionResult<IEnumerable<ClusterChange>>> GetClusterChangesBySymbol(string symbol)
        {
            var clusterChanges = await _context.ClusterChanges
                .Where(c => c.Symbol == symbol)
                .Select(c => new {
                    c.Id,
                    c.CurrencyId,
                    c.Symbol,
                    c.FromClusterId,
                    c.ToClusterId,
                    c.ChangeTimestamp
                })
                .OrderBy(c => c.ChangeTimestamp)
                .ToListAsync();

            if (!clusterChanges.Any())
            {
                return NotFound(new { message = $"No cluster changes found for symbol: {symbol}" });
            }

            return Ok(clusterChanges);
        }
    }
}