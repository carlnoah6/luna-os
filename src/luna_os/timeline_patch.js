// Add edge numbering - insert after drawing each arrow path

// After: svg.appendChild(path);
// Add:

// Add edge label (from->to)
const edgeLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
const pathLength = path.getTotalLength();
const midPoint = path.getPointAtLength(pathLength / 2);
edgeLabel.setAttribute('x', midPoint.x);
edgeLabel.setAttribute('y', midPoint.y - 5);
edgeLabel.setAttribute('text-anchor', 'middle');
edgeLabel.setAttribute('font-size', '10');
edgeLabel.setAttribute('font-weight', '700');
edgeLabel.setAttribute('fill', '#666');
edgeLabel.setAttribute('style', 'background: white; padding: 2px;');
edgeLabel.textContent = `${depId}→${s.id}`;
labelSvg.appendChild(edgeLabel);
