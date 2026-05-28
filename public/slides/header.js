<script src='https://cdnjs.cloudflare.com/ajax/libs/jquery/3.3.1/jquery.min.js'></script>
<script src='https://cdnjs.cloudflare.com/ajax/libs/jquery-zoom/1.7.21/jquery.zoom.min.js'></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pangu/4.0.7/pangu.js"></script>

<script>
(function () {
  $('.hover_large img')
    .wrap('<span style="display:inline-block"></span>')
    .css('display', 'block')
    .parent()
    .zoom();
})();
</script>
<script>
pangu.spacingElementByClassName('pangujs');
</script>