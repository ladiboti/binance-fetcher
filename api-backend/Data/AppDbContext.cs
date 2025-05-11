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
        public DbSet<ClusterChange> ClusterChanges { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<Currency>().ToTable("currencies");
            modelBuilder.Entity<ClusterChange>().ToTable("cluster_changes");
            
            base.OnModelCreating(modelBuilder);
        }
    }
}
