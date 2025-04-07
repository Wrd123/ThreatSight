// static/js/grid_interaction.js

// Get all grid items
const gridItems = document.querySelectorAll('.grid-item');
const overlay = document.getElementById('overlay');

// Add click event to each grid item
gridItems.forEach(item => {
    item.addEventListener('click', function() {
        // Toggle expanded class
        if (this.classList.contains('expanded')) {
            this.classList.remove('expanded');
            overlay.style.display = 'none';
        } else {
            // Remove expanded class from any other items
            gridItems.forEach(otherItem => {
                otherItem.classList.remove('expanded');
            });
            
            // Add expanded class to this item
            this.classList.add('expanded');
            overlay.style.display = 'block';
        }
    });
});

// Close expanded view when clicking overlay
overlay.addEventListener('click', function() {
    gridItems.forEach(item => {
        item.classList.remove('expanded');
    });
    overlay.style.display = 'none';
});