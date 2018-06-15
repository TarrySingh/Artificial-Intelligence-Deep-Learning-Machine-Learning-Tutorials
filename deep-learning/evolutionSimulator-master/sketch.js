let person;
let boundary;
let creatures = []

const Render = Matter.Render
const engine = Matter.Engine.create();
const world = engine.world;
const generationPeriod = 20;
let generation = new Generation(25);
let settled = false;

function setup() {
	let canvas = createCanvas(windowWidth * 0.95, windowHeight * 0.95);
	frameRate(60);
	rectMode(CENTER);
	textSize(18)
	fill(255);

	// Initialize Generation
	generation.initialize(Person);
	generation.species.forEach((creature) => { creature.add_to_world(world) });

	// Boundary
	boundary = new SimpleBoundary();
	boundary.add_to_world();

	// Mouse Constraint
	let canvasMouse = Matter.Mouse.create(canvas.elt);
	canvasMouse.pixelRatio = pixelDensity();
	let m = Matter.MouseConstraint.create(engine, { mouse: canvasMouse });
	Matter.World.add(world, m);

	// Restart Generation after 5 seconds
	setInterval(() => {
		generation.evolve();
		console.log(generation.avg_score);
		settled = false;
	}, generationPeriod * 1000);

	// Run the renderer
	// let render = Render.create({
	// 	element: document.body,
	// 	engine: engine,
	// 	options: {
	// 		height, width
	// 	}
	// })
	// Render.run(render);

	// let renderMouse = Matter.Mouse.create(render.canvas);
	// renderMouse.pixelRatio = pixelDensity();
	// Matter.World.add(world, Matter.MouseConstraint.create(engine, {
	// 	mouse: renderMouse
	// }));
}

let counter = 1;
function draw() {
	if (counter >= 60) {
		counter = 0;
		settled = true;
	}
	counter++;
	background(color(15, 15, 19));

	// Display Boundary
	boundary.display();

	// Display Creatures
	generation.species.forEach((creature) => {
		creature.show();
		creature.adjust_score();
		if (counter % 4 === 0 && settled) {
			creature.think(boundary);
		}
	});

	// Display Stats
	textSize(18)
	fill("red");
	text("Generation: " + generation.generation, 40, 50);
	text("HighScore: " + generation.high_score.toFixed(2), 40, 70);
	text("Average Score " + generation.avg_score.toFixed(2), 40, 90);
	text("Population: " + generation.population, 40, 110);
	text("Generation Period: " + generationPeriod + " seconds", 40, 130);
	text("Mutation Rate: " + 5 + "%", 40, 150);
	text("Progress: " + generation.progress.toFixed(2), 40, 170);
	
	// Display Inheritance
	textSize(14);
	fill('green');
	text("Creature\tParentA\t\tParentB", width * 0.78, 40)
	generation.species.forEach((creature, index) => {
		let txt = '';
		if (creature.parents.length !== 0)
			txt = `${creature.id} \t\t\t ${creature.parents[0].id} (${creature.parents[0].score.toFixed(0)}) \t\t\t ${creature.parents[1].id}(${creature.parents[1].score.toFixed(0)})`;
		else
			txt = `${creature.id} \t\t\t ------ \t\t\t ------`
		text(txt, width * 0.80, 60 + (15 * index));
	})

	// Run Matter-JS Engine
	Matter.Engine.update(engine);
}