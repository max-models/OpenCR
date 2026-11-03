// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/OpenCR/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'OpenCR',
	title: 'OpenCR',
	description: 'Collisional-radiative models of low-temperature plasmas in Python.',
	pythonPackage: 'opencr',
	// the method cards: a Numerics section with the overview (numerics/index.mdx) and one page per
	// module, in the order of the model, and a Numerics section on each method's API page
	methodcard: {
		pages: true,
		index: false,
		labels: {
			'opencr.kinetics.system': 'Population equations',
			'opencr.core.rates': 'Rate models',
			'opencr.kinetics.rates': 'Rate coefficients',
			'opencr.kinetics.eedf': 'Electron energy distributions',
			'opencr.kinetics.effective': 'Quasi-static reduction',
			'opencr.solvers.steady': 'Steady states',
			'opencr.solvers.diagnostics': 'Convergence checks',
			'opencr.solvers.transient': 'Time integration',
			'opencr.kinetics.radiation': 'Radiation trapping',
			'opencr.model': 'Model solves',
			'opencr.analysis.emission': 'Emission',
			'opencr.analysis.results': 'Photon emissivity',
			'opencr.analysis.quantities': 'Electron cooling',
			'opencr.models.hydrogen_atomic': 'Atomic hydrogen',
			'opencr.data.hydrogen': 'Hydrogen data',
			'opencr.constants': 'Constants and units',
			'opencr.inference.sensitivity': 'Sensitivities',
			'opencr.tables': 'Effective-rate tables',
			'opencr.backends.batched': 'Batched reduction',
			'opencr.backends.jax_backend': 'JAX backend',
		},
	},
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Guides', link: '/guides/model/' },
		{ label: 'Tutorials', link: '/tutorials/' },
		{ label: 'Full examples', link: '/full-examples/' },
		{ label: 'Numerics', link: '/numerics/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/opencr/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
			],
		},
		{
			label: 'Guides',
			items: [
				{ label: 'The CR model', slug: 'guides/model' },
				{ label: 'Atomic hydrogen', slug: 'guides/atomic-hydrogen' },
				{ label: 'Molecular hydrogen', slug: 'guides/molecular-hydrogen' },
				{ label: 'Argon and metals', slug: 'guides/argon-and-metals' },
				{ label: 'Sensitivities and inference', slug: 'guides/inference' },
				{ label: 'Data and provenance', slug: 'guides/data' },
				{ label: 'Solvers and troubleshooting', slug: 'guides/solvers' },
				{ label: 'Verification', slug: 'guides/verification' },
				{ label: 'Backends', slug: 'guides/backends' },
				{ label: 'Benchmarks', slug: 'guides/benchmarks' },
			],
		},
		{
			label: 'Tutorials',
			items: [{ autogenerate: { directory: 'tutorials' } }],
		},
		{
			label: 'Full examples',
			items: [{ autogenerate: { directory: 'full-examples' } }],
		},
		{
			label: 'Development',
			items: [
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
			],
		},
	],
});
