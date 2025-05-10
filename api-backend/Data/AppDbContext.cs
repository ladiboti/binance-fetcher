using api_backend.Models;
using Microsoft.EntityFrameworkCore;

namespace api_backend.Data
{
    public class AppDbContext : DbContext
    {
        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options)
        {
        }

        public DbSet<Currency> Currencies { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<Currency>().ToTable("currencies");
            
            base.OnModelCreating(modelBuilder);
        }
    }
}
