// grid_interaction.js - updated to focus on content expansion

document.addEventListener('DOMContentLoaded', function() {
    // Get all grid items
    const gridItems = document.querySelectorAll('.grid-item');
    const overlay = document.getElementById('overlay');
    
    // Create a container for expanded content
    const expandedContent = document.createElement('div');
    expandedContent.className = 'expanded-content';
    expandedContent.style.display = 'none';
    document.body.appendChild(expandedContent);
    
    // Add close button to expanded content
    const closeButton = document.createElement('span');
    closeButton.innerHTML = '&times;';
    closeButton.className = 'close-button';
    expandedContent.appendChild(closeButton);
    
    // Function to close expanded content
    function closeExpandedContent() {
        expandedContent.style.display = 'none';
        overlay.style.display = 'none';
    }
    
    // Close button event
    closeButton.addEventListener('click', closeExpandedContent);
    
    // Overlay click event
    overlay.addEventListener('click', closeExpandedContent);
    
    // Add click event to each grid item
    gridItems.forEach(item => {
        item.addEventListener('click', function() {
            // Get the title
            const title = this.querySelector('h3').textContent;
            
            // Create a title element for the expanded content
            const titleElement = document.createElement('h3');
            titleElement.textContent = title;
            
            // Get the main content (image or table)
            let contentElement;
            if (this.querySelector('img')) {
                contentElement = this.querySelector('img').cloneNode(true);
                contentElement.style.maxWidth = '90%';
                contentElement.style.maxHeight = '80vh';
                contentElement.style.margin = '0 auto';
                contentElement.style.display = 'block';
            } else if (this.querySelector('table')) {
                contentElement = this.querySelector('table').cloneNode(true);
                contentElement.style.width = '90%';
                contentElement.style.margin = '0 auto';
            } else if (this.querySelector('div')) {
                contentElement = this.querySelector('div').cloneNode(true);
                contentElement.style.width = '90%';
                contentElement.style.margin = '0 auto';
            }
            
            // Clear and update expanded content
            expandedContent.innerHTML = '';
            expandedContent.appendChild(closeButton);
            expandedContent.appendChild(titleElement);
            
            if (contentElement) {
                expandedContent.appendChild(contentElement);
            }
            
            // Show expanded content and overlay
            expandedContent.style.display = 'block';
            overlay.style.display = 'block';
        });
    });
});